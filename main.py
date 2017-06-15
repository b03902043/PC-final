from models import *
from layer_utils import *
import os, sys


# Freeze weights in the discriminator for stacked training
def make_trainable(net, val):
	net.trainable = val
	for l in net.layers:
		l.trainable = val

output_folder = 'output'

cycle_gan = CycleGAN(num_features=64, shape=(128, 128, 3), bch_img_num=100, task_name='summer2winter_yosemite')
try:
	cycle_gan.fit(epoch_num=200, disc_iter=20, save_period=1, pic_dir=output_folder)
	cycle_gan.save(path=output_folder)
except KeyboardInterrupt:
	K.clear_session()
	try:
		sys.exit(0)
	except SystemExit:
		os._exit(0)
else:
	K.clear_session()