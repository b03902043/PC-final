from models import *
import tensorflow as tf

def disc_loss(y_true, y_pred):
	return (tf.reduce_mean(tf.square(self.fake_pool_rec_A)) + tf.reduce_mean(tf.squared_difference(self.rec_A, 1)))/2.0

# Freeze weights in the discriminator for stacked training
def make_trainable(net, val):
	net.trainable = val
	for l in net.layers:
		l.trainable = val

cycle_gan = CycleGAN()
cycle_gan.setup_model()
cycle_gan.cyc_A.compile(optimizer=cycle_gan.gopt, loss=disc_loss)