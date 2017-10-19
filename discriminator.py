import os;
import math;
import numpy as np;
import tensorflow as tf;
from nn import *;

class Discriminator(object):
	def __init__(self,params,phase):
		self.phase = phase;
		self.batch_size = params.batch_size if phase=='train' else 1;
		self.batch_norm = params.batch_norm;
		self.image_shape = [params.image_shape,params.image_shape,3];
		print('Building the Discriminator.......');

	def run(self,images,train,reuse=False):
		image_shape = self.image_shape;
		bn = self.batch_norm;
		with tf.variable_scope('discriminator') as scope:
			if reuse:
				scope.reuse_variables();
			conv1 = convolution(images,5,5,64,2,2,'d_conv1');
			conv1 = batch_norm(conv1,'d_bn1',train,bn,'relu');
			conv2 = convolution(conv1,5,5,128,2,2,'d_conv2');
			conv2 = batch_norm(conv2,'d_bn2',train,bn,'relu');
			conv3 = convolution(conv2,5,5,256,2,2,'d_conv3');
			conv3 = batch_norm(conv3,'d_bn3',train,bn,'relu');
			conv4 = convolution(conv3,5,5,512,2,2,'d_conv4');
			conv4 = batch_norm(conv4,'d_bn4',train,bn,'relu');
			feats = tf.reshape(conv4,[self.batch_size,-1]);
			output = fully_connected(feats,1,'d_fc1');
			return nonlinear(output,'sigmoid');
			