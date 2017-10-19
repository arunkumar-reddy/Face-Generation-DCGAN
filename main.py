import sys;
import os;
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3';

import tensorflow as tf;
import argparse;
from model import *;
from dataset import *;

def main(argv):
	parser = argparse.ArgumentParser();
	'''Model Parameters'''
	parser.add_argument('--phase', default='train', help='Train or Test');
	parser.add_argument('--vector_size', default=100, help='Latent vector dimensioms');
	parser.add_argument('--image_shape', default=64,help ='Size of the input image used for training');
	parser.add_argument('--output_shape', default=64, help='Size of the image to be generated');
	parser.add_argument('--train_dataset', default='/home/arun/Datasets/CelebA/', help='Directory containing the training images');
	parser.add_argument('--train_dir', default='./train/', help='Directory to store training results');
	parser.add_argument('--test_samples', default=30, help='Number of test samples');
	parser.add_argument('--test_dir', default='./test/', help='Directory to store testing images');
	parser.add_argument('--save_dir', default='./models/', help='Directory to contain the trained model');
	parser.add_argument('--save_period', type=int, default=2000, help='Period to save the trained model');
	'''Hyper parameters'''
	parser.add_argument('--solver', default='Adam', help='Gradient Descent Optimizer to use: Can be adam, momentum, rmsprop or sgd') 
	parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs');
	parser.add_argument('--batch_size', type=int, default=64, help='Batch size');
	parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate');
	parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay');
	parser.add_argument('--momentum', type=float, default=0.9, help='Momentum (for some optimizers)'); 
	parser.add_argument('--decay', type=float, default=0.9, help='Decay (for some optimizers)'); 
	parser.add_argument('--batch_norm', action='store_true', default=False, help='Turn on to use batch normalization');

	args = parser.parse_args();
	with tf.Session() as sess:
		if(args.phase=='train'):
			data = train_data(args);
			model = Model(args,'train');
			sess.run(tf.global_variables_initializer());
			model.Train(sess,data);
		else:
			model = Model(args,'test');
			sess.run(tf.global_variables_initializer());
			model.load(sess);
			model.Test(sess);

main(sys.argv);