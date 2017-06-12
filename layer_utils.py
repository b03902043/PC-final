from keras.models import Model, Sequential, model_from_json
from keras.layers import Input, Flatten, Dropout, Embedding, Dense, Activation, Merge
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Add, Concatenate, Dot
from keras.utils import plot_model
import keras.backend as K
from keras.initializers import TruncatedNormal, Constant
import os
import numpy as np
from PIL import Image

def build_resnet_block(i_res, dim, name="resnet", res_layer_num=2):
	res_out = Conv2D(dim, (3, 3), activation='relu', strides=(1, 1), padding='same', 
		kernel_initializer=TruncatedNormal(stddev=0.02), bias_initializer=Constant(0.0))(i_res)
	for i in range(res_layer_num-1):
		res_out = Conv2D(dim, (3, 3), strides=(1, 1), padding='same',
			kernel_initializer=TruncatedNormal(stddev=0.02), bias_initializer=Constant(0.0))(res_out)
	res_out = Add()([res_out, i_res])
	res_out = Activation('relu')(res_out)
	return res_out

def randReadImg(tag, num, shp=(256, 256, 3), printName=False):
	fo = 'input/apple2orange/train{}'.format(tag)
	fs = [f for f in os.listdir(fo) if os.path.isfile(os.path.join(fo, f))]
	ret = np.zeros(shape=((num, ) + shp))
	for _ in range(num):
		rd = np.random.randint(len(fs))
		if printName:
			print (fs[rd])
		img = Image.open(os.path.join(fo, fs[rd]))
		ret[_] = np.asarray(img)
	return ret
