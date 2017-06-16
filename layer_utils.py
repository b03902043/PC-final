from keras.models import Model, Sequential, model_from_json
from keras.layers import Input, Flatten, Dropout, Embedding, Dense, Activation, Merge
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
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
import tensorflow as tf

class InstanceNormalization2D(Layer):
	''' Thanks for github.com/jayanthkoushik/neural-style '''
	def __init__(self, **kwargs):
		super(InstanceNormalization2D, self).__init__(**kwargs)

	def build(self, input_shape):
		self.scale = self.add_weight(name='scale', shape=(input_shape[1],), 
			initializer=TruncatedNormal(mean=1.0, stddev=0.02), trainable=True)
		self.shift = self.add_weight(name='shift', shape=(input_shape[1],), 
			initializer=Constant(0.0), trainable=True)
		super(InstanceNormalization2D, self).build(input_shape)

	def call(self, x, mask=None):
		epsilon = 1e-5

		def image_expand(tensor):
			return K.expand_dims(K.expand_dims(tensor, -1), -1)

		def batch_image_expand(tensor):
			return image_expand(K.expand_dims(tensor, 0))

		hw = K.cast(x.shape[2] * x.shape[3], K.floatx())
		mu = K.sum(x, [-1, -2]) / hw
		mu_vec = image_expand(mu) 
		sig2 = K.sum(K.square(x - mu_vec), [-1, -2]) / hw
		y = (x - mu_vec) / (K.sqrt(image_expand(sig2) + epsilon))

		scale = batch_image_expand(self.scale)
		shift = batch_image_expand(self.shift)
		return scale*y + shift 

	def compute_output_shape(self, input_shape):
		return input_shape 

class ReflectPadding2D(Layer):
	def __init__(self, padding = 1, **kwargs):
		self.padding = padding
		super(ReflectPadding2D, self).__init__(**kwargs)

	def build(self, input_shape):
		super(ReflectPadding2D, self).build(input_shape)

	def call(self, x, mask=None):
		w, h = K.int_shape(x)[1:3]
		left, right, top, down = x[:, :, 0], x[:, :, -1], x[:, 0], x[:, -1]
		x = ZeroPadding2D((self.padding, self.padding))(x)
		return x

	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[1]+2*self.padding, input_shape[2]+2*self.padding, input_shape[3])

class DeConv2D(Conv2DTranspose):
	def get_output_shape_for_(self):
		pass

def conv2d(tensor, num_feature, kernel_shape=(3, 3), strides=(1, 1), padding='valid',
	kernel_std=0.02, bias_init=0.0, up_sample=1, norm=True, relu=True, relu_alpha=0.0 ):
	if up_sample > 1:
		nn = UpSampling2D(size=(up_sample, up_sample))(tensor)
	else:
		nn = tensor

	# pad = int(np.floor(kernel_shape[0] / 2))
	# nn = ZeroPadding2D(padding=(pad, pad))(nn)
	nn = Conv2D(num_feature, kernel_shape, strides=strides, padding=padding,
		kernel_initializer=TruncatedNormal(stddev=kernel_std), bias_initializer=Constant(bias_init))(nn)

	if norm:
		nn = InstanceNormalization2D()(nn)
		
	if relu:
		nn = LeakyReLU(relu_alpha)(nn)

	return nn

def build_resnet_block(i_res, dim):
	# res_out = tf.pad(i_res, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
	res_out = ZeroPadding2D((1, 1))(i_res)
	res_out = conv2d(res_out, dim)
	# res_out = tf.pad(res_out, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
	res_out = ZeroPadding2D((1, 1))(res_out)
	res_out = conv2d(res_out, dim, relu=False)
	res_out = Add()([res_out, i_res])
	return res_out

def sharpen(data):
	if len(data.shape) == 3:
		data = data.reshape((1, ) + data.shape)
	return data

def randReadImg(tag, num, shp=(256, 256, 3), printName=False, task_name = 'apple2orange'):
	fo = 'input/{}/train{}'.format(task_name, tag)
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
		fig = plt.figure(figsize=(sub_w, sub_h))
		for i in range(num):
			ax = fig.add_subplot(sub_h, sub_w, i+1)
			ax.set_axis_off()
			ax.imshow(imgs[i])
		plt.savefig(path, dpi=300)
		plt.clf()

def simple_save(path, obj):
	with open(path, 'wb') as f:
		pickle.dump(f, obj)
