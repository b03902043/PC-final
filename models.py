from keras.models import Model, Sequential, model_from_json
from keras.layers import Input, Flatten, Dropout, Embedding, Dense, Activation, Merge
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Add, Concatenate, Dot
from keras.utils import plot_model
import keras.backend as K
from keras.initializers import TruncatedNormal, Constant
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam
from PIL import Image
import numpy as np

from layer_utils import *

adam = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)

def scheduler(epoch):
	lr_base = 0.0002
	if epoch > 100:
		return lr_base * (200 - epoch)
	return lr_base

class Generator:

	def __init__(self, num_features=64, img_w=256, img_h=256, img_l=3, name='generator'):
		self.nf = num_features
		self.img_size = (img_w, img_h, img_l)
		self.model = self.build_model(needSum=False)
		self.name = name

	def build_model(self, needSum = True):
		input_gen = Input(shape=self.img_size)
		nn = Conv2D(self.nf, (7, 7), activation='relu', strides=(1, 1), padding='same')(input_gen)
		nn = BatchNormalization()(nn)
		nn = Conv2D(self.nf*2, (3, 3), activation='relu', strides=(2, 2), padding='same')(nn)
		nn = BatchNormalization()(nn)
		nn = Conv2D(self.nf*4, (3, 3), activation='relu', strides=(2, 2), padding='same')(nn)
		nn = BatchNormalization()(nn)

		# transform
		nn = build_resnet_block(nn, 64*4)
		nn = build_resnet_block(nn, 64*4)
		nn = build_resnet_block(nn, 64*4)
		nn = build_resnet_block(nn, 64*4)
		nn = build_resnet_block(nn, 64*4)
		nn = build_resnet_block(nn, 64*4)

		# decoding
		nn = Conv2DTranspose(self.nf*2, (3, 3), activation='relu', strides=(2, 2), padding='same')(nn)
		nn = BatchNormalization()(nn)
		nn = Conv2DTranspose(self.nf, (3, 3), activation='relu', strides=(2, 2), padding='same')(nn)
		nn = BatchNormalization()(nn)
		nn = Conv2D(3, (7, 7), activation='relu', strides=(1, 1), padding='same')(nn)
		gen = BatchNormalization()(nn)
		
		generator = Model(inputs=input_gen, outputs=gen)
		generator.compile(loss='binary_crossentropy', optimizer=adam)
		if needSum:
			generator.summary()
		# plot_model(generator, to_file='gen.png')

		return generator

	def connect(self, next_networks):
		if not hasattr(next_networks, 'model'):
			raise 'In {}, argument \"next_model\" does not have model attribute '.format(self.__class__.__name__)
		return Model(inputs=self.model.input, outputs=next_networks.model(self.model.output))

class Discriminator:
	def __init__(self, num_features=64, img_w=256, img_h=256, img_l=3, name='discriminator'):
		self.nf = num_features
		self.img_size = (img_w, img_h, img_l)
		self.model = self.build_model(needSum=False)
		self.name = name

	def build_model(self, needSum = True):
		filter_w = 4
		input_dis = Input(shape=self.img_size)
		nn = Conv2D(self.nf, (filter_w, filter_w), strides=(2, 2), padding='same',
			kernel_initializer=TruncatedNormal(stddev=0.02), bias_initializer=Constant(0.0))(input_dis)
		nn = LeakyReLU(0.2)(nn)
		nn = Conv2D(self.nf*2, (filter_w, filter_w), strides=(2, 2), padding='same',
			kernel_initializer=TruncatedNormal(stddev=0.02), bias_initializer=Constant(0.0))(nn)
		nn = LeakyReLU(0.2)(nn)
		nn = BatchNormalization()(nn)
		nn = Conv2D(self.nf*4, (filter_w, filter_w), strides=(2, 2), padding='same',
			kernel_initializer=TruncatedNormal(stddev=0.02), bias_initializer=Constant(0.0))(nn)
		nn = LeakyReLU(0.2)(nn)
		nn = BatchNormalization()(nn)
		nn = Conv2D(self.nf*8, (filter_w, filter_w), strides=(1, 1), padding='same',
			kernel_initializer=TruncatedNormal(stddev=0.02), bias_initializer=Constant(0.0))(nn)
		nn = LeakyReLU(0.2)(nn)
		nn = BatchNormalization()(nn)
		dis = Conv2D(1, (filter_w, filter_w), strides=(1, 1), padding='same',
			kernel_initializer=TruncatedNormal(stddev=0.02), bias_initializer=Constant(0.0))(nn)
		discriminator = Model(inputs=input_dis, outputs=dis)
		discriminator.compile(loss=, optimizer=adam)
		if needSum:
			discriminator.summary()
		return discriminator

	def connect(self, next_networks):
		if not hasattr(next_networks, 'model'):
			raise 'In {}, argument \"next_model\" does not have model attribute '.format(self.__class__.__name__)
		return Model(inputs=self.model.input, outputs=next_networks.model(self.model.output))


class CycleGAN:
	def __init__(self):
		# print 'Init CycleGAN'
		self.gopt = Adam(lr=0.0002, beta_1=0.5)
		self.dopt = Adam(lr=0.0002, beta_1=0.5)
		self.setup_model()

	def setup_input(self, A = None, B = None):
		img_num = 30
		self.inputA = randReadImg('A', img_num)
		self.inputB = randReadImg('B', img_num)

	'''
		The function setup the model for training
		genB/ genA : Generated images by corresponding generator of input_A and input_B
		cyc_A/ cyc_B : Images generated after feeding genA/genB to corresponding generator. This is use to calcualte cyclic loss
	'''

	def setup_model(self):
		# if self.inputA is None or self.inputB is None:
		# 	raise Exception('Input must be assigned before setup model')

		self.genB = Generator(name='GenA2B')
		self.genA = Generator(name='GenB2A')
		self.clf_A = Discriminator(name='ClfA')
		self.clf_B = Discriminator(name='ClfB')

		self.clf_gen = self.genA.connect(Discriminator(name='clfA'))
		self.clf_gen = self.genB.connect(Discriminator(name='clfB'))
		self.cyc_A = self.genB.connect(Generator(name='GenB2A'))
		self.cyc_B = self.genA.connect(Generator(name='GenA2B'))

		# plot_model(self.cyc_A, to_file='cycA.png')
		self.cyc_A.summary()
		# self.cyc_A.compile(optimizer=opt)

	

	# def __call__(self, input):
	# 	self.
		# Build GAN model


# gan_input = Input(shape=[100])
# H = generator(gan_input)
# gan_V = discriminator(H)
# GAN = Model(gan_input, gan_V)
# GAN.compile(loss='categorical_crossentropy', optimizer=opt)
# GAN.summary()

# 		dec_gen_A = build_discriminator(gen_A, "discriminator_A")
# 		dec_gen_B = build_discriminator(gen_B, "discriminator_B")
# 		cyc_A = build_generator(gen_B, "generator_BtoA")
# 		cyc_B = build_generator(gen_A, "generator_AtoB")






