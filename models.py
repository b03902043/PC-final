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
import tensorflow as tf

from layer_utils import *

lambda_gan = 10
batch_size = 1
pool_size = 50
adam = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)

def disc_loss(y_true, y_pred):
	return (tf.reduce_mean(tf.square(y_pred)) + tf.reduce_mean(tf.squared_difference(y_true, 1)))/2.0

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
		# generator.compile(loss='binary_crossentropy', optimizer=adam)
		if needSum:
			generator.summary()
		# plot_model(generator, to_file='gen.png')

		return generator

	def connect(self, next_networks):
		if not hasattr(next_networks, 'model'):
			raise 'In {}, argument \"next_model\" does not have model attribute '.format(self.__class__.__name__)
		return Model(inputs=self.model.input, outputs=next_networks.model(self.model.output))

	def predict(self, **kwargs):
		return self.model.predict(**kwargs)

	def fit(self, **kwargs):
		return self.model.fit(**kwargs)

	def compile(self, **kwargs):
		return self.model.compile(**kwargs)

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
		# discriminator.compile(loss=, optimizer=adam)
		if needSum:
			discriminator.summary()
		return discriminator

	def connect(self, next_networks):
		if not hasattr(next_networks, 'model'):
			raise 'In {}, argument \"next_model\" does not have model attribute '.format(self.__class__.__name__)
		return Model(inputs=self.model.input, outputs=next_networks.model(self.model.output))

	def predict(self, **kwargs):
		return self.model.predict(**kwargs)

	def fit(self, **kwargs):
		return self.model.fit(**kwargs)

	def compile(self, **kwargs):
		return self.model.compile(**kwargs)

class CycleGAN:

	def __init__(self, shape = (256, 256, 3), dis_iter = 32):
		# print 'Init CycleGAN'
		self.shp = shape
		self.gopt = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)
		self.dopt = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)
		self.fake_images_A, self.fake_num_A = np.zeros((pool_size, 1,) + shape), 0
		self.fake_images_B, self.fake_num_B = np.zeros((pool_size, 1,) + shape), 0
		self.d_iter = dis_iter
		self.setup_model()

	def collect_images(self, A = None, B = None):
		self.batch_img_num = 10
		self.inputA = randReadImg('A', self.batch_img_num)
		self.inputB = randReadImg('B', self.batch_img_num)

	'''
		The function setup the model for training
		genB/ genA : Generated images by corresponding generator of input_A and input_B
		cyc_A/ cyc_B : Images generated after feeding genA/genB to corresponding generator. This is use to calcualte cyclic loss
	'''
	def setup_model(self):
		# if not hasattr(self, 'inputA') or not hasattr(self, 'inputB'):
		# 	raise Exception('Input must be assigned before setup model')

		self.genB = Generator(name='GenA2B')
		self.genA = Generator(name='GenB2A')
		self.clf_A = Discriminator(name='clf_A')
		self.clf_B = Discriminator(name='clf_B')

		self.clf_genA = self.genA.connect(Discriminator(name='clf_A'))
		self.clf_genB = self.genB.connect(Discriminator(name='clf_B'))
		self.cyc_A = self.genB.connect(Generator(name='GenB2A'))
		self.cyc_B = self.genA.connect(Generator(name='GenA2B'))

		# plot_model(self.cyc_A, to_file='cycA.png')
		self.cyc_A.summary()
		# self.cyc_A.compile(optimizer=opt)
		self.trainnerG = Model([ self.genB.model.input, self.genA.model.input ],
			[self.clf_genB.output, self.clf_genA.output, self.cyc_A.output, self.cyc_B.output])
		self.trainnerG.compile(optimizer=self.gopt, 
			loss=['MSE', 'MSE', 'MAE', 'MAE'], 
			loss_weight=[1, 1, lambda_gan, lambda_gan])

		self.real_A, self.real_B, self.fake_A, self.fake_B = Input(self.shp), Input(self.shp), Input(self.shp), Input(self.shp)
		self.clf_real_A, self.clf_real_B, self.clf_fake_A, self.clf_fake_B = self.clf_A.model(self.real_A),	\
			self.clf_B.model(self.real_B), self.clf_A.model(self.fake_A), self.clf_B.model(self.fake_B)

		self.trainnerD = Model([self.real_A, self.fake_A, self.real_B, self.fake_B], 
			[self.clf_real_A, self.clf_fake_A, self.clf_real_B, self.clf_fake_B])
		self.trainnerD.compile(optimizer=self.dopt, loss='MSE')

	def fit(self, epoch_num = 10):
		for i in range(epoch_num):

			self.collect_images()

			A_fake = self.update_fake_pool(self.fake_images_A, self.genA.predict(x=self.inputB), self.fake_num_A)
			B_fake = self.update_fake_pool(self.fake_images_B, self.genB.predict(x=self.inputA), self.fake_num_B)

			ones  = np.ones((self.batch_img_num,) + self.trainnerG.output_shape[0][1:])
			zeros = np.zeros((self.batch_img_num, ) + self.trainnerG.output_shape[0][1:])

			# train discriminator
			for _ in range(self.d_iter):
				_, rA_dloss, fA_dloss, rB_dloss, fB_dloss = self.trainnerD.train_on_batch([self.inputA, A_fake, self.inputB, B_fake], 
					[zeros, ones * 0.9, zeros, ones * 0.9])	# label given (assign real=0, fake=0.9)

			# train generator
			# Target "zero" represent the classifier assume A -gen-> B -clf-> B (100%)
			# That is, generator can cheat clf
			_, rA_gloss, fA_gloss, rB_gloss, fB_gloss = self.trainnerG.train_on_batch([self.inputA, self.inputB],
				[zeros, zeros, self.inputA, self.inputB])	
			
			print ('Generator Loss:')
			pritn ('Real A: {}, Fake A: {}, Real B: {}, Fake B: {}'.format(rA_gloss, fA_gloss, rB_gloss, fB_gloss))

			print ('Discriminator Loss:')
			pritn ('Real A: {}, Fake A: {}, Real B: {}, Fake B: {}'.format(rA_dloss, fA_dloss, rB_dloss, fB_dloss))

			print ('Discriminator A (accuracy) : real({}), fake({})'.format(
				self.clf_A.predict(x=self.inputA).mean(), self.clf_A.predict(x=A_fake).mean()))

			self.fake_num_A += self.batch_img_num
			self.fake_num_B += self.batch_img_num

	def update_fake_pool(self, fake_pool, new_fakes, num_fakes):
		'''
			This function saves the gen images to A/B image pools.
			The return value represent the gen images to training
		'''
		overfull_num = pool_size - (num_fakes + len(new_fakes) - 1)
		
		if overfull_num > 0:
			fake_pool = np.roll(fake_pool, -overfull_num)
			ret_images = fake_pool[-len(new_fakes):].copy()	# this part can be random swap-in return images
			fake_pool[-len(new_fakes):] = new_fakes
		else:
			fake_pool[num_fakes : num_fakes + len(new_fakes) - 1] = new_fakes

		return fake_pool[np.random.choice(num_fakes - overfull_num, size=len(new_fakes), replace=False)]

