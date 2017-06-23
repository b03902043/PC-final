from keras.models import Model, Sequential, model_from_json
from keras.layers import Input, Flatten, Dropout, Embedding, Dense, Activation, Merge, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose, ZeroPadding2D
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
import pickle as pkl
import os, sys, gc
from keras.models import load_model

from layer_utils import *

lambda_gan = 10
batch_size = 1
adam = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)
i_layerid, o_layerid = 0, 0

def disc_loss(y_true, y_pred):
	return (tf.reduce_mean(tf.square(y_pred)) + tf.reduce_mean(tf.squared_difference(y_true, 1)))/2.0

def scheduler(epoch):
	lr_base = 0.0002
	if epoch > 100:
		return lr_base * (200 - epoch)
	return lr_base

def init_network(model):
	for w in model.weights:
		if w.name.startswith('conv2d') and w.name.endswith('kernel'):
			value = np.random.normal(loc=0.0, scale=0.02, size=w.get_value().shape)
			w.set_value(value.astype('float32'))
		if w.name.startswith('conv2d') and w.name.endswith('bias'):
			value = np.zeros(w.get_value().shape)
			w.set_value(value.astype('float32'))
	return model

class Generator:

	def __init__(self, num_features=64, im_shape = (256, 256, 3), name='generator', Cmodel = None, res_cnt = 6):
		self.nf = num_features
		self.img_size = im_shape
		self.name = name
		self.input_num = 0	# input_num
		self.model = self.build_model(res_cnt) if Cmodel is None else Cmodel

	def build_model(self, res_cnt, needSum = False):
		input_gen = Input(shape=self.img_size)

		# encoding
		nn = ZeroPadding2D((3, 3))(input_gen)
		nn = conv2d(nn, self.nf, (7, 7), strides=(1, 1))
		nn = conv2d(nn, self.nf*2, (3, 3), strides=(2, 2), padding='same')
		nn = conv2d(nn, self.nf*4, (3, 3), strides=(2, 2), padding='same')

		# transform
		for i in range(res_cnt):
			nn = build_resnet_block(nn, self.nf*4)

		# decoding
		nn = conv2d(nn, self.nf*2, (3, 3), strides=(1, 1), up_sample=2, padding='same')
		nn = conv2d(nn, self.nf, (3, 3), strides=(1, 1), up_sample=2, padding='same')
		if res_cnt == 6:
			nn = ZeroPadding2D((3, 3))(nn)
		nn = conv2d(nn, 3, (7, 7), strides=(1, 1), norm=False, relu=False, padding='valid' if res_cnt == 6 else 'same')
		gen = Activation('tanh')(nn)

		generator = Model(inputs=input_gen, outputs=gen)
		if needSum:
			generator.summary()

		return generator

	def connect(self, next_networks, input_id = 0):
		if not hasattr(next_networks, 'model'):
			raise 'In {}, argument \"next_model\" does not have model attribute '.format(self.__class__.__name__)
		# return Generator(Cmodel=Model(inputs=self.model.input, outputs=next_networks.model(self.model.output)))
		# if self.input_num < input_id:
			# raise 'Input out of bound error'
		# next_networks.input_num += 1
		return next_networks.model(self.model.output)
		# return Model(inputs=self.model.get_input_at(input_id), outputs=next_networks.model(self.model.output))
		# return Model(inputs=Input(shape=self.img_size), outputs=next_networks.model(self.model.output))

	def predict(self, X):
		return self.model.predict(X)

	def fit(self, **kwargs):
		return self.model.fit(**kwargs)

	def compile(self, **kwargs):
		return self.model.compile(**kwargs)

	def save(self, path, with_img=False, show_shapes=True):
		self.model.save_weights(path+'.h5')
		if with_img:
			plot_model(self.model, to_file=path+'.png', show_shapes=show_shapes)

	def load(self, path, **kwargs):
		self.model.load_weights(path, **kwargs)

class Discriminator:
	def __init__(self, num_features=64, im_shape = (256, 256, 3), name='discriminator', Cmodel = None):
		self.nf = num_features
		self.img_size = im_shape
		self.name = name
		self.input_num = 0	# input_num
		self.model = self.build_model() if Cmodel is None else Cmodel

	def build_model(self, needSum = False, needSigmoid = False):
		filter_w = 4
		input_dis = Input(shape=self.img_size)

		nn = conv2d(input_dis, self.nf, (filter_w, filter_w), strides=(2, 2), padding='same',
			kernel_std=0.02, bias_init=0.0, relu_alpha=0.2)

		nn = conv2d(nn, self.nf*2, (filter_w, filter_w), strides=(2, 2), padding='same',
			kernel_std=0.02, bias_init=0.0, relu_alpha=0.2)

		nn = conv2d(nn, self.nf*4, (filter_w, filter_w), strides=(2, 2), padding='same',
			kernel_std=0.02, bias_init=0.0, relu_alpha=0.2)

		nn = conv2d(nn, self.nf*8, (filter_w, filter_w), strides=(1, 1), padding='same',
			kernel_std=0.02, bias_init=0.0, relu_alpha=0.2)

		nn = conv2d(nn, 1, (filter_w, filter_w), strides=(1, 1), padding='same',
			kernel_std=0.02, bias_init=0.0, relu=False, norm=False)

		# nn = Conv2D(self.nf, (filter_w, filter_w), strides=(2, 2), padding='same',
		# 	kernel_initializer=TruncatedNormal(stddev=0.02), bias_initializer=Constant(0.0))(input_dis)
		# nn = LeakyReLU(0.2)(nn)

		# nn = Conv2D(self.nf*2, (filter_w, filter_w), strides=(2, 2), padding='same',
		# 	kernel_initializer=TruncatedNormal(stddev=0.02), bias_initializer=Constant(0.0))(nn)
		# nn = InstanceNormalization2D()(nn)
		# nn = LeakyReLU(0.2)(nn)

		# nn = Conv2D(self.nf*4, (filter_w, filter_w), strides=(2, 2), padding='same',
		# 	kernel_initializer=TruncatedNormal(stddev=0.02), bias_initializer=Constant(0.0))(nn)
		# nn = InstanceNormalization2D()(nn)
		# nn = LeakyReLU(0.2)(nn)

		# nn = Conv2D(self.nf*8, (filter_w, filter_w), strides=(1, 1), padding='same',
		# 	kernel_initializer=TruncatedNormal(stddev=0.02), bias_initializer=Constant(0.0))(nn)
		# nn = InstanceNormalization2D()(nn)
		# nn = LeakyReLU(0.2)(nn)

		# nn = Conv2D(1, (filter_w, filter_w), strides=(1, 1), padding='same',
		# 	kernel_initializer=TruncatedNormal(stddev=0.02), bias_initializer=Constant(0.0))(nn)
		
		if needSigmoid:
			nn = Activation('sigmoid')(nn)

		discriminator = Model(inputs=input_dis, outputs=nn)
		# discriminator.compile(loss=, optimizer=adam)
		if needSum:
			discriminator.summary()
		return discriminator

	def connect(self, next_networks, input_id = 0):
		if not hasattr(next_networks, 'model'):
			raise 'In {}, argument \"next_model\" does not have model attribute '.format(self.__class__.__name__)
		# return Discriminator(Cmodel=Model(inputs=self.model.input, outputs=next_networks.model(self.model.output)))
		# next_networks.conn[0] += 1
		# i_layerid = self.conn[0]
		# o_layerid = self.conn[1] - 1
		# self.conn[1] += 1
		# if self.input_num < input_id:
		# 	raise 'Input out of bound error'
		# next_networks.input_num += 1
		return next_networks.model(self.model.output)
		# return Model(inputs=self.model.get_input_at(input_id), outputs=next_networks.model(self.model.output))
		# return Model(inputs=self.model.get_input_at(input_id), outputs=next_networks.model(self.model.output))

	def predict(self, X):
		return self.model.predict(X)

	def fit(self, **kwargs):
		return self.model.fit(**kwargs)

	def compile(self, **kwargs):
		return self.model.compile(**kwargs)

	def save(self, path, with_img=False, show_shapes=True):
		self.model.save_weights(path+'.h5')
		if with_img:
			plot_model(self.model, to_file=path+'.png', show_shapes=show_shapes)

	def load(self, path, **kwargs):
		self.model.load_weights(path, **kwargs)

class CycleGAN:

	def __init__(self, ngf = 32, ndf = 64, shape = (256, 256, 3), bch_img_num = 10, ps = 50, task_name='apple2orange', pic_dir=None):
		# print 'Init CycleGAN'
		self.shp = shape
		self.ngf = ngf
		self.ndf = ndf
		self.gopt = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)
		self.dopt = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)
		self.batch_img_num = bch_img_num
		self.pool_size = ps if bch_img_num < ps else bch_img_num
		self.fake_images_A, self.fake_num_A = np.zeros((self.pool_size, ) + shape), 0
		self.fake_images_B, self.fake_num_B = np.zeros((self.pool_size, ) + shape), 0
		self.task_name = task_name
		self.pic_dir = pic_dir
		if self.pic_dir is not None:
			self.check_dir()
		self.setup_model()

	def collect_images(self, A = None, B = None):
		self.inputA = randReadImg('A', self.batch_img_num, shp = self.shp, task_name=self.task_name) / 127.5 - 1
		self.inputB = randReadImg('B', self.batch_img_num, shp = self.shp, task_name=self.task_name) / 127.5 - 1

	def check_dir(self):
		if not os.path.isdir(self.pic_dir):
			os.makedirs(self.pic_dir)
		self.pic_dir = os.path.join(self.pic_dir, self.task_name)
		if not os.path.isdir(self.pic_dir):
			os.makedirs(self.pic_dir)

	'''
		The function setup the model for training
		genB/ genA : Generated images by corresponding generator of input_A and input_B
		cyc_A/ cyc_B : Images generated after feeding genA/genB to corresponding generator. This is use to calcualte cyclic loss
	'''
	def setup_model(self):
		# if not hasattr(self, 'inputA') or not hasattr(self, 'inputB'):
		# 	raise Exception('Input must be assigned before setup model')

		self.genB = Generator(name='GenA2B', im_shape=self.shp, num_features=self.ngf, res_cnt=6 if self.shp[0] == 128 else 9)
		self.genA = Generator(name='GenB2A', im_shape=self.shp, num_features=self.ngf, res_cnt=6 if self.shp[0] == 128 else 9)
		self.clf_A = Discriminator(name='clf_A', im_shape=self.shp, num_features=self.ndf)	# clf input0 = real, input1 = fake
		self.clf_B = Discriminator(name='clf_B', im_shape=self.shp, num_features=self.ndf)

		self.genA.model = init_network(self.genA.model)
		self.genB.model = init_network(self.genB.model)
		self.clf_A.model = init_network(self.clf_A.model)
		self.clf_B.model = init_network(self.clf_B.model)

		self.realA, self.realB = Input(self.shp), Input(self.shp)
		self.fakeA, self.fakeB = self.genA.model(self.realB), self.genB.model(self.realA)

		self.clf_genA = self.clf_A.model(self.fakeA)
		self.clf_genB = self.clf_B.model(self.fakeB)

		self.cyc_A = self.genA.model(self.fakeB)
		self.cyc_B = self.genB.model(self.fakeA)	

		# plot_model(self.cyc_A, to_file='cycA.png')
		# self.cyc_A.summary()
		# self.cyc_A.compile(optimizer=opt)
		self.trainnerG = Model([ self.realA, self.realB ],
			[self.clf_genA, self.cyc_A, self.clf_genB, self.cyc_B])
		self.trainnerG.compile(optimizer=self.gopt, 
			loss=['MSE', 'MAE', 'MSE', 'MAE'], 
			loss_weights=[1, lambda_gan, 1, lambda_gan])

		realA, realB = Input(self.shp), Input(self.shp)
		fakeA, fakeB = Input(self.shp), Input(self.shp)

		clf_fakeA = self.clf_A.model(fakeA)
		clf_fakeB = self.clf_B.model(fakeB)
		clf_realA = self.clf_A.model(realA)
		clf_realB = self.clf_B.model(realB)

		# self.real_A, self.real_B, self.fake_A, self.fake_B = Input(self.shp), Input(self.shp), Input(self.shp), Input(self.shp)
		# self.clf_real_A, self.clf_real_B, self.clf_fake_A, self.clf_fake_B = self.clf_A.model(self.real_A),	\
		# 	self.clf_B.model(self.real_B), self.clf_A.model(self.fake_A), self.clf_B.model(self.fake_B)

		self.trainnerD = Model([ realA, fakeA, realB, fakeB ], 
			[clf_realA, clf_fakeA, clf_realB, clf_fakeB])
		self.trainnerD.compile(optimizer=self.dopt, loss='MSE')

	def fit(self, epoch_num = 10, disc_iter = 10, save_period = 1):

		for i in range(epoch_num):
			print ('Epoch {}'.format(i+1))

			# change learning rate
			self.UpdateOptimizerLR(i+1, [self.trainnerG, self.trainnerD])

			self.collect_images()

			A_fake = self.update_fake_pool(self.fake_images_A, self.genA.predict(self.inputB), self.fake_num_A)
			B_fake = self.update_fake_pool(self.fake_images_B, self.genB.predict(self.inputA), self.fake_num_B)

			ones  = np.ones((self.batch_img_num,) + self.trainnerG.output_shape[0][1:])
			zeros = np.zeros((self.batch_img_num, ) + self.trainnerG.output_shape[0][1:])

			print ('Training discriminator')
			# train discriminator
			for _ in range(disc_iter):
				_, rA_dloss, fA_dloss, rB_dloss, fB_dloss = self.trainnerD.train_on_batch([self.inputA, A_fake, self.inputB, B_fake],
					[zeros, ones, zeros, ones])	# label given (assign real=0, fake=0.9)

			# train generator
			# Target "zero" represent the classifier assume A -gen-> B -clf-> B (100%)
			# That is, generator can cheat clf
			print ('Training generator')
			_, rA_gloss, fA_gloss, rB_gloss, fB_gloss = self.trainnerG.train_on_batch([self.inputA, self.inputB],
				[zeros, self.inputA, zeros, self.inputB])	
			
			print ('Generator Loss:')
			print ('Real A: {}, Fake A: {}, Real B: {}, Fake B: {}'.format(rA_gloss, fA_gloss, rB_gloss, fB_gloss))

			print ('Discriminator Loss:')
			print ('Real A: {}, Fake A: {}, Real B: {}, Fake B: {}'.format(rA_dloss, fA_dloss, rB_dloss, fB_dloss))

			print ('Discriminator A (accuracy) : real({}), fake({})'.format(
				self.clf_A.predict(self.inputA).mean(), self.clf_A.predict(A_fake).mean()))

			sys.stdout.flush()

			if (i+1) % save_period == 0 and self.pic_dir is not None:

				max_width = 10

				ImageA = self.inputA[:max_width]  # N * W * W * l
				ImageA2B = self.genB.predict(ImageA)
				ImageA2B2A = self.genA.predict(ImageA2B)
				ImageB = self.inputB[:max_width]  # 1 * N * W * W * l
				ImageB2A = self.genA.predict(ImageB)
				ImageB2A2B = self.genB.predict(ImageB2A)

				ImageA2B = sharpen(ImageA2B)
				ImageA2B2A = sharpen(ImageA2B2A)
				ImageB2A = sharpen(ImageB2A)
				ImageB2A2B = sharpen(ImageB2A2B)

				# np.save(os.path.join(pic_dir, 'ia2b'+str(i)), ImageA2B)
				# np.save(os.path.join(pic_dir, 'ib2a'+str(i)), ImageB2A)

				Imgs = ( np.r_[ ImageA, ImageA2B, ImageA2B2A, ImageB, ImageB2A, ImageB2A2B ] + 1 ) * 0.5
				saveImg(Imgs, sub_w = len(ImageA), path = os.path.join(self.pic_dir, '{}.jpg'.format(i)))
				del Imgs, ImageA2B, ImageA2B2A, ImageB2A, ImageB2A2B
			del self.inputA, self.inputB, ones, zeros, A_fake, B_fake
			gc.collect()

			self.fake_num_A += self.batch_img_num
			self.fake_num_B += self.batch_img_num

	def update_fake_pool(self, fake_pool, new_fakes, num_fakes):
		'''
			This function saves the gen images to A/B image pools.
			The return value represent the gen images to training
		'''
		overfull_num = (num_fakes + len(new_fakes)) - self.pool_size
		
		if overfull_num > 0:
			fake_pool = np.roll(fake_pool, -overfull_num)
			ret_images = fake_pool[-len(new_fakes):].copy()	# this part can be random swap-in return images
			fake_pool[-len(new_fakes):] = new_fakes
			return fake_pool[np.random.choice(self.pool_size, size=len(new_fakes), replace=False)]
		else:
			fake_pool[num_fakes : num_fakes + len(new_fakes)] = new_fakes

		return new_fakes
		
	def UpdateOptimizerLR(self, ep, mds):
		lr_base = 0.0002
		if ep > 200:
			lr_base = 0
		elif ep > 100:
			lr_base *= (200-ep)/100.0
		else:
			return 

		for md in mds:
			K.set_value(md.optimizer.lr, lr_base)

	def save(self, path = None, with_img = False, show_shapes = False, withParam = True):
		path = self.pic_dir if path is None else path
		rpath = path + '-models'

		if not os.path.isdir(rpath):
			os.makedirs(rpath)

		if withParam:
			self.genB.save(os.path.join(rpath, 'a2b'), with_img=with_img, show_shapes=show_shapes)
			self.genA.save(os.path.join(rpath, 'b2a'), with_img=with_img, show_shapes=show_shapes)
			self.clf_A.save(os.path.join(rpath, 'clfA'), with_img=with_img, show_shapes=show_shapes)
			self.clf_B.save(os.path.join(rpath, 'clfB'), with_img=with_img, show_shapes=show_shapes)
			self.trainnerG.save_weights(os.path.join(rpath, 'trainnerG.h5'))
			self.trainnerD.save_weights(os.path.join(rpath, 'trainnerD.h5'))

		if with_img:
			plot_model(self.genA.model, to_file=os.path.join(rpath, 'genA.png'), show_shapes=show_shapes)
			plot_model(self.clf_A.model, to_file=os.path.join(rpath, 'clfA.png'), show_shapes=show_shapes)
			plot_model(self.trainnerG, to_file=os.path.join(rpath, 'trainnerG.png'), show_shapes=show_shapes)
			plot_model(self.trainnerD, to_file=os.path.join(rpath, 'trainnerD.png'), show_shapes=show_shapes)

