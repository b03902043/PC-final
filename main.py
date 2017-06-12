from models import *
from layer_utils import *
import os, sys


# Freeze weights in the discriminator for stacked training
def make_trainable(net, val):
	net.trainable = val
	for l in net.layers:
		l.trainable = val

cycle_gan = CycleGAN()
try:
	cycle_gan.fit(epoch_num=10)
except KeyboardInterrupt:
	K.clear_session()
	try:
		sys.exit(0)
	except SystemExit:
		os._exit(0)
else:
	K.clear_session()