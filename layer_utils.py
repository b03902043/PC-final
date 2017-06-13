from keras.models import Model, Sequential, model_from_json
from keras.layers import Input, Flatten, Dropout, Embedding, Dense, Activation, Merge
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Add, Concatenate, Dot
from keras.utils import plot_model
import keras.backend as K
from keras.initializers import TruncatedNormal, Constant
import os, pickle
import numpy as np
import PIL
from PIL import Image
import matplotlib as mpl
try:
	mpl.use('Agg')
except:
	try:
		mpl.use('TkAgg')
	except:
		print ('Your platform cannot use matplotlib')
		HasMatPlotLib = False
	else:
		HasMatPlotLib = True
else:
	HasMatPlotLib = True

import matplotlib.pyplot as plt

from keras.engine.topology import Layer

class InstanceNormalization2D(Layer):
	''' Thanks for github.com/jayanthkoushik/neural-style '''
	def __init__(self, **kwargs):
		super(InstanceNormalization2D, self).__init__(**kwargs)

	def build(self, input_shape):
		self.scale = self.add_weight(name='scale', shape=(input_shape[1],), initializer="one", trainable=True)
		self.shift = self.add_weight(name='shift', shape=(input_shape[1],), initializer="zero", trainable=True)
		super(InstanceNormalization2D, self).build(input_shape)

	def call(self, x, mask=None):
		def image_expand(tensor):
			return K.expand_dims(K.expand_dims(tensor, -1), -1)

		def batch_image_expand(tensor):
			return image_expand(K.expand_dims(tensor, 0))

		hw = K.cast(x.shape[2] * x.shape[3], K.floatx())
		mu = K.sum(x, [-1, -2]) / hw
		mu_vec = image_expand(mu) 
		sig2 = K.sum(K.square(x - mu_vec), [-1, -2]) / hw
		y = (x - mu_vec) / (K.sqrt(image_expand(sig2)) + K.epsilon())

		scale = batch_image_expand(self.scale)
		shift = batch_image_expand(self.shift)
		return scale*y + shift 
#	   else:
#		   raise NotImplemented("Please complete `CycGAN/layers/padding.py` to run on backend {}.".format(K.backend()))

	def compute_output_shape(self, input_shape):
		return input_shape 

class DeConv2D(Conv2DTranspose):
	def get_output_shape_for_(self):
		pass

def deconv2d(tensor, n_feature, kernel_shape, strides=(1, 1), padding='valid'):
	ret = UpSampling2D(strides)(tensor)
	ret = Conv2D(n_feature, kernel_shape, padding=padding)(ret)
	return ret


def build_resnet_block(i_res, dim, name="resnet", res_layer_num=2):
	res_out = ZeroPadding2D()(i_res)
	res_out = Conv2D(dim, (3, 3), strides=(1, 1), padding='valid', 
		kernel_initializer=TruncatedNormal(stddev=0.02), bias_initializer=Constant(0.0))(res_out)
	res_out = InstanceNormalization2D()(res_out)

	for i in range(res_layer_num - 1):
		res_out = Activation('relu')(res_out)
		res_out = ZeroPadding2D()(res_out)
		res_out = Conv2D(dim, (3, 3), strides=(1, 1), padding='valid',
			kernel_initializer=TruncatedNormal(stddev=0.02), bias_initializer=Constant(0.0))(res_out)
		res_out = InstanceNormalization2D()(res_out)

	res_out = Add()([res_out, i_res])
	return res_out

def sharpen(data):
	if len(data.shape) == 3:
		data = data.reshape((1, ) + data.shape)
	return data

def randReadImg(tag, num, shp=(256, 256, 3), printName=False):
	fo = 'input/apple2orange/train{}'.format(tag)
	fs = [f for f in os.listdir(fo) if os.path.isfile(os.path.join(fo, f))]
	ret = np.zeros(shape=((num, ) + shp))
	for _ in range(num):
		rd = np.random.randint(len(fs))
		if printName:
			print (fs[rd])
		img = Image.open(os.path.join(fo, fs[rd]))
		img = img.resize((shp[0], shp[1]), PIL.Image.ANTIALIAS)
		ret[_] = np.asarray(img)
	return ret

def saveImg(imgs, sub_w = 5, path = None):

	num = len(imgs)
	sub_h = (num+sub_w-1) / sub_w

	if path is not None and HasMatPlotLib:
		# save
		fig = plt.figure()
		for i in range(num):
			ax = fig.add_subplot(sub_h, sub_w, i+1)
			ax.set_axis_off()
			ax.imshow(imgs[i])
		plt.savefig(path)
		plt.clf()

def simple_save(path, obj):
	with open(path, 'wb') as f:
		pickle.dump(f, obj)
