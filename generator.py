import os;
import math;
import numpy as np;
import tensorflow as tf;
from nn import *;

class Generator(object):
	def __init__(self,params,phase):
		self.phase = phase;
		self.batch_size = params.batch_size if phase=='train' else 1;
		self.batch_norm = params.batch_norm;
		self.image_shape = [params.image_shape,params.image_shape,3];
		self.output_shape = [params.output_shape,params.output_shape,3];
		print('Building the Generator......');

	def deconv_shape(self,size,stride):
		return int(math.ceil(float(size)/float(stride)));

	def run(self,vectors,train,reuse=False):
		if(self.phase=='train'):
			output_shape = self.image_shape;
		else:
			output_shape = self.output_shape;
		deconv3_shape = self.deconv_shape(output_shape[0],2);
		deconv2_shape = self.deconv_shape(deconv3_shape,2);
		deconv1_shape = self.deconv_shape(deconv2_shape,2);
		feat_shape = self.deconv_shape(deconv1_shape,2);
		bn = self.batch_norm;
		with tf.variable_scope('generator') as scope:
			if reuse:
				scope.reuse_variables();
			feats = fully_connected(vectors,1024*feat_shape*feat_shape,'g_fc1');
			feats = tf.reshape(feats,[self.batch_size,feat_shape,feat_shape,1024]);
			feats = batch_norm(feats,'g_bn1',train,bn,'relu');
			deconv1 = deconvolution(feats,[self.batch_size,deconv1_shape,deconv1_shape,512],5,5,2,2,'g_deconv1');
			deconv1 = batch_norm(deconv1,'g_bn2',train,bn,'relu');
			deconv2 = deconvolution(deconv1,[self.batch_size,deconv2_shape,deconv2_shape,256],5,5,2,2,'g_deconv2');
			deconv2 = batch_norm(deconv2,'g_bn3',train,bn,'relu');
			deconv3 = deconvolution(deconv2,[self.batch_size,deconv3_shape,deconv3_shape,128],5,5,2,2,'g_deconv3');
			deconv3 = batch_norm(deconv3,'g_bn2',train,bn,'relu');
			output = deconvolution(deconv3,[self.batch_size,output_shape[0],output_shape[1],3],5,5,2,2,'g_deconv4');
			return nonlinear(output,'tanh');