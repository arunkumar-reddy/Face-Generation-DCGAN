import os;
import sys;
import numpy as np;
import tensorflow as tf;
import matplotlib.pyplot as plt;

from tqdm import tqdm;
from skimage.io import imread,imsave,imshow;
from skimage.transform import resize;
from dataset import *;
from generator import *;
from discriminator import *;

class Model(object):
	def __init__(self,params,phase):
		self.params = params;
		self.phase = phase;
		self.vector_size = params.vector_size;
		self.batch_size = params.batch_size if phase=='train' else 1;
		self.image_shape = [params.image_shape,params.image_shape,3];
		self.output_shape = [params.output_shape,params.output_shape,3];
		self.save_dir = os.path.join(params.save_dir,self.params.solver+'/');
		self.global_step = tf.Variable(0,name='global_step',trainable=False);
		self.saver = tf.train.Saver(max_to_keep = 100);
		self.build();

	def build(self):
		print('Building the Model......');
		image_shape = self.image_shape;
		images = tf.placeholder(tf.float32,[self.batch_size]+image_shape);
		vectors = tf.placeholder(tf.float32,[self.batch_size,self.vector_size]);
		train = tf.placeholder(tf.bool);
		reuse = False if self.phase =='train' else True;
		discriminator = Discriminator(self.params,self.phase);
		generator = Generator(self.params,self.phase);
		output = generator.run(vectors,train,reuse);
		real = discriminator.run(images,train,reuse);
		fake = discriminator.run(output,train,reuse=True);
		real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real,labels=tf.ones_like(real)));
		fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake,labels=tf.zeros_like(fake)));
		gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake,labels=tf.ones_like(fake)));
		disc_loss = real_loss+fake_loss;

		self.images = images;
		self.vectors = vectors;
		self.train = train;
		self.disc_loss = disc_loss;
		self.gen_loss = gen_loss;
		self.real_loss = real_loss;
		self.fake_loss = fake_loss;
		self.output = output;

		if self.params.solver == 'adam':
			disc_solver = tf.train.AdamOptimizer(self.params.learning_rate);
			gen_solver = tf.train.AdamOptimizer(self.params.learning_rate);
		elif self.params.solver == 'momentum':
			disc_solver = tf.train.MomentumOptimizer(self.params.learning_rate,self.params.momentum);
			gen_solver = tf.train.MomentumOptimizer(self.params.learning_rate,self.params.momentum);
		elif self.params.solver == 'rmsprop':
			disc_solver = tf.train.RMSPropOptimizer(self.params.learning_rate,self.params.weight_decay,self.params.momentum);
			gen_solver = tf.train.RMSPropOptimizer(self.params.learning_rate,self.params.weight_decay,self.params.momentum);
		else:
			disc_solver = tf.train.GradientDescentOptimizer(self.params.learning_rate);
			gen_solver = tf.train.GradientDescentOptimizer(self.params.learning_rate);

		tensorflow_variables = tf.trainable_variables();
		disc_variables = [variable for variable in tensorflow_variables if 'd_' in variable.name];
		gen_variables = [variable for variable in tensorflow_variables if 'g_' in variable.name];
		disc_gradients,_ = tf.clip_by_global_norm(tf.gradients(self.disc_loss,disc_variables),3.0);
		gen_gradients,_ = tf.clip_by_global_norm(tf.gradients(self.gen_loss,gen_variables),3.0);
		disc_optimizer = disc_solver.apply_gradients(zip(disc_gradients,disc_variables));
		gen_optimizer = gen_solver.apply_gradients(zip(gen_gradients,gen_variables),global_step=self.global_step);
		self.disc_optimizer = disc_optimizer;
		self.gen_optimizer = gen_optimizer;
		print('Model built......');

	def Train(self,sess,data):
		print('Training the Model......');
		epochs = self.params.epochs;
		for epoch in tqdm(list(range(epochs)),desc='Epoch'):
			for i in tqdm(list(range(data.batches)),desc='Batch'):
				files = data.next_batch();
				images = self.load_images(files);
				vectors = np.random.uniform(-1,1,size=(self.batch_size,self.vector_size));
				disc_loss,_ = sess.run([self.disc_loss, self.disc_optimizer],feed_dict={self.images:images, self.train:True, self.vectors:vectors});
				'''For each update to the discriminator, perform two updates to the generator to prevent early convergence of the discriminator...'''
				for i in range(2):
					global_step,gen_loss,_ = sess.run([self.global_step,self.gen_loss,self.gen_optimizer],feed_dict={self.train:True, self.vectors:vectors});
				print(' Discriminator_loss = %f Generator_loss = %f'%(disc_loss,gen_loss));
				if(global_step%5000==0):
					output = sess.run(self.output,feed_dict={self.vectors:vectors, self.train:False});
					self.save_image(output[0],'train_sample_'+str(global_step));
				if(global_step%self.params.save_period==0):
					self.save(sess);
			data.reset();
		self.save(sess);
		print('Model trained......');

	def Test(self,sess):
		print('Testing the Model......');
		result_dir = self.params.test_result;
		for i in tqdm(list(range(self.params.test_samples)),desc='Batch'):
			vector = np.random.uniform(-1,1,size=(self.vector_size));
			output = sess.run(self.output,feed_dict={self.vectors:vector, self.train:False});
			self.save_image(output,'test_sample_'+str(i+1));
		print('Testing completed......');

	def save(self,sess):
		print(('Saving model to %s......'% self.save_dir));
		self.saver.save(sess,self.save_dir,self.generator_step);

	def load(self,sess):
		print('Loading model.....');
		checkpoint = tf.train.get_checkpoint_state(self.save_dir);
		if checkpoint is None:
			print("Error: No saved model found. Please train first...");
			sys.exit(0);
		self.saver.restore(sess, checkpoint.model_checkpoint_path);

	def load_images(self,files):
		images = [];
		image_shape = self.image_shape;
		for image_file in files:
			image = imread(image_file);
			image = resize(image,(image_shape[0],image_shape[1]));
			image = (image-127.5)/127.5;
			images.append(image);
		images = np.array(images,np.float32);
		return images;

	def save_image(self,output,name):
		output = (output*127.5)+127.5;
		if(self.phase=='train'):
			file_name = self.params.train_dir+name+'.png';
		else:
			file_name = self.params.test_dir+name+'.png';
		imsave(file_name,output);
		print('Saving the image %s...',file_name);