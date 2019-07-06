# Jan/12
# DCL baseline: gru based
# Step 1: y -> y -> y, history -> real -> eval
# Step 2: x -> y -> y
# Step 3: x -> x -> y (2, 3 are opposite)
# Step 4: y -> x -> y

import numpy as np 
from random import seed
seed(1)
import random
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping

import sys
import utils
import build_model
from read_data import load_default_data, load_pm25_data, load_script_data, load_KDD_2018_data


import tensorflow as tf
import keras.backend as K
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
sess = tf.Session(config=config)
K.set_session(sess)


def train_rnn(x_train, x_valid, is_under_train, is_under_valid, 
				real_train, real_valid, eval_train, eval_valid,  
				is_select_normal = False, window_size = 24, units = 256, layers = 1, debug = False): 

	print('rnn')
	real_train = real_train[:, -1]
	real_valid = real_valid[:, -1]
	eval_train = eval_train[:, -1]
	eval_valid = eval_valid[:, -1]

	if is_select_normal == False: 
		x_train = x_train[:, -window_size-1:-1]
		x_valid = x_valid[:, -window_size-1:-1]
	else: 
		from baseline import select_normal
		x_train = select_normal(x_train[:, :-1], is_under_train[:, :-1], interval) 
		x_valid = select_normal(x_valid[:, :-1], is_under_valid[:, :-1], interval)
	x_train = np.reshape(x_train, [x_train.shape[0], x_train.shape[1], 1])
	x_valid = np.reshape(x_valid, [x_valid.shape[0], x_valid.shape[1], 1])
	print('x (window) =', x_train.shape, x_valid.shape)

	rnn_model = build_model.rnn(max_length = window_size)
	earlyStopping = EarlyStopping(monitor = 'val_loss', patience = 20)
	rnn_model.fit(x_train, real_train, validation_data = (x_valid, real_valid), 
					epochs = 1000, batch_size = 1024, verbose = 2, shuffle = False, 
					callbacks = [earlyStopping])
	
	if debug: 
		pred = rnn_model.predict(x_train)
		utils.get_error(real_train, pred[:, 0], '(train, x), int=' + str(window_size))
		utils.get_error(eval_train, pred[:, 0], '(train, y), int=' + str(window_size))
		pred = rnn_model.predict(x_valid)
		utils.get_error(real_valid, pred[:, 0], '(valid, x), int=' + str(window_size))
		utils.get_error(eval_valid, pred[:, 0], '(valid, y), int=' + str(window_size))
	else:
		pred = rnn_model.predict(x_valid)
		utils.get_error(eval_valid, pred[:, 0], '(valid, y), int=' + str(window_size))

	print('real =', real_valid[:10])
	print('eval =', eval_valid[:10])
	print('pred =', pred[:10, 0])
	print('mean', np.mean(real_valid), np.mean(eval_valid), np.mean(pred))
	print('shape =', real_valid.shape, eval_valid.shape, pred.shape)


if __name__ == '__main__':
	#ts_train, ts_valid, ts_under_train, ts_under_valid, is_under_train, is_under_valid, ts_features_train, ts_features_valid = load_data(True)
	#ts_train, ts_valid, ts_under_train, ts_under_valid, is_under_train, is_under_valid = load_default_data()
	#ts_train, ts_valid, ts_under_train, ts_under_valid, is_under_train, is_under_valid = load_pm25_data('prob77')
	#ts_train, ts_valid, ts_under_train, ts_under_valid, is_under_train, is_under_valid = load_taxi_data()
	#ts_train, ts_valid, ts_under_train, ts_under_valid, is_under_train, is_under_valid = load_script_data()
	ts_train, ts_valid, ts_under_train, ts_under_valid, is_under_train, is_under_valid = load_KDD_2018_data(sys.argv[2])
	
	# yyy
	fout = open(sys.argv[4], 'a')
	print('filename =', sys.argv[4])
	fout.write('yyy\t')
	fout.close()
	train_rnn(ts_train, ts_valid, is_under_train, is_under_valid, ts_train, ts_valid, ts_train, ts_valid)
	
	# xxy
	fout = open(sys.argv[4], 'a')
	fout.write('xxy\t')
	fout.close()
	train_rnn(ts_under_train, ts_under_valid, is_under_train, is_under_valid, ts_under_train, ts_under_valid, ts_train, ts_valid)
	
	# xyy
	#train_rnn(ts_under_train, ts_under_valid, is_under_train, is_under_valid, ts_train, ts_valid, ts_train, ts_valid)
	# yxy
	#train_rnn(ts_train, ts_valid, is_under_train, is_under_valid, ts_under_train, ts_under_valid, ts_train, ts_valid)
	
	
