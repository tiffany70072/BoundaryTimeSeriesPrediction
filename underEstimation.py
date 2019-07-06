import numpy as np
import sys
sys.path.insert(0, '../')

import util
from sklearn.utils import shuffle
from read_data import load_KDD_2018_data

import keras.backend as K
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
tf_session = tf.Session(config=config)
K.set_session(tf_session)

def set_nan(ts, ts_under):
	is_win = ts_under >= ts # if under < ts: ts = nan
	ts_nan = np.copy(ts)
	ts_nan[np.invert(is_win)] = np.nan # if not win, we can only know the under value
	return ts_nan

def uniform(ts, is_under):
	uni_ratio = np.random.uniform(0., 1., ts.shape)
	ts_under = np.where(is_under, ts*uni_ratio, ts)
	print(np.mean(ts_under), np.mean(ts))
	return ts_under


class UnderEstimation(object):
	def __init__(self):
		self.model_type = 'rnn'
		self.rebuild = int(sys.argv[6])	# use rebuild method for x
		self.debug = 0 	# write file, no terminal information
		
		self.epoch = 300
		self.interval = 24

		self.loss_list = ['yes', 'diff', 'mse', 'mse_sigma']
		self.loss = 'mse'
		if len(sys.argv) > 2: self.loss = sys.argv[2]
		self.gru_type_list = ['gru', 'mygru', 'close', 
							'dgru', 'dgru_reset', 'dgru_update', 'dgru_bias', 'dgru_update_bias', 
							'gru2w', 'gru2w_reset', 'gru2w_all', 'gru2w_update', 'gru2w_bias', 'gru2w_update_bias']
		self.gru_type = 'gru' 
		if len(sys.argv) > 3: self.gru_type = sys.argv[3]
		
		self.message = sys.argv[1] + " " + self.gru_type + " " + self.loss + " " + str(self.interval) + " rebuild=" + str(self.rebuild) + "\t"
		print(self.message)
		if len(sys.argv) > 4: 
			fout = open(sys.argv[4], 'a')
			print('filename =', sys.argv[4])
			fout.write(self.message)
			fout.close()
		
		ts_train, ts_valid, ts_under_train, ts_under_valid, is_under_train, is_under_valid = load_KDD_2018_data(sys.argv[5])
		ts_nan_train = set_nan(ts_train, ts_under_train)
		ts_nan_valid = set_nan(ts_valid, ts_under_valid)

		self.all_mean = np.nanmean(np.where(is_under_train, np.nan, ts_under_train))
		print('all_mean = %.3f' %self.all_mean)

		# default
		self.x_train, self.wp_bp_train, self.is_under_train, self.ts_train = self.prepare_data(ts_train, ts_nan_train, ts_under_train, is_under_train)
		self.x_valid, self.wp_bp_valid, self.is_under_valid, self.ts_valid = self.prepare_data(ts_valid, ts_nan_valid, ts_under_valid, is_under_valid)
		
		self.x_train_origin = np.copy(self.x_train) # original under value, with shape = (None, interval * 2)
		self.x_valid_origin = np.copy(self.x_valid)

	def prepare_data(self, ts, ts_nan, ts_under, is_under):
		wp_bp = np.column_stack((ts_nan[:, -1], ts_under[:, -1])) # change later to whole ts, check features[index]
		
		ts_under = ts_under[:, -self.interval*2-1:-1]
		is_under = is_under[:, -self.interval*2-1:-1]

		x0 = np.copy(np.expand_dims(ts_under, axis = 2))
		x1 = np.copy(np.expand_dims(is_under, axis = 2))
		x = np.concatenate([x1, x0], axis = 2)

		return x, wp_bp, is_under, ts[:, -self.interval-1:]

	def pad_average(self, ts_under, is_under):
		count1, count2 = 0, 0
		for i in range(ts_under.shape[0]):
			if np.sum(is_under[i]) != is_under.shape[1]: # not all is_under
				row_mean = np.nanmean(np.where(is_under[i], np.nan, ts_under[i, :, 2]))
				ts_under[i, :self.interval:, 2] = np.where(is_under[i, :self.interval], row_mean, ts_under[i, :self.interval, 2])	
				count1 += np.sum(is_under[i, :self.interval])
			else:
				ts_under[i, :self.interval:, 2] = np.where(is_under[i, :self.interval], self.all_mean, ts_under[i, :self.interval, 2])
				count2 += np.sum(is_under[i, :self.interval])
		print('count =', count1, count2, ts_under.shape[0]*self.interval)
		return ts_under

	def rebuild_one_cycle_rnn(self, ts_under, is_under, ts_origin, ts): # only operate without features
		ts_new = np.copy(ts_under[:, :, 1:2])
		for i in range(self.interval): 
			x = ts_under[:, i:i+self.interval] 			
			y = self.model.predict(x, batch_size = 1024)[:, 0] # 0: normal, 1: under
			ts_new[:, i+self.interval, 0] = np.where(is_under[:, i], y, ts_under[:, i, 1])
			
		is_win_pred = (ts_new[:, self.interval:, 0] > ts_under[:, self.interval:, 0]) * is_under[:, self.interval:]
		ts_new[:, self.interval:, 0] = np.where(is_win_pred, ts_new[:, self.interval:, 0], ts_origin[:, self.interval:, 1]) # method 2
		print('predict larger (origin) =', np.sum(is_win_pred)/float(np.sum(is_under[:, self.interval:])))
		print('new =', ts_new[0, -19:, 0].astype(int))
		print('ori =', ts_origin[0, -19:, 1].astype(int))
		print('rea =', ts[0, -20:].astype(int))
		print('error from under:', np.mean(np.abs(ts_under[:, self.interval:, 0] - ts[:, :-1])), ', from rebuild:', np.mean(np.abs(ts_new[:, self.interval:, 0] - ts[:, :-1])))
		return ts_new

	def my_early_stop(self, val_loss, min_val_loss, epo, min_epoch):
		print('epoch = %d, val_loss = %.3f, min_epoch = %d, min_loss = %.3f' %(epo, val_loss, min_epoch, min_val_loss))
		if val_loss < min_val_loss:
			min_val_loss = val_loss
			min_epoch = epo
		if epo - min_epoch > 20: 
			print('early stop! epoch = %d, val_loss = %.3f, min_val_loss = %.3f' %(epo, val_loss, min_val_loss))
			return True, min_val_loss, min_epoch
		return False, min_val_loss, min_epoch

	def train(self):
		from loss.get_loss import get_loss_yes, get_loss_sigma, get_loss_lognormal, get_loss_diff, get_loss_mse, get_loss_mse_sigma
		from get_model import get_linear_model, get_mlp, get_rnn, get_two_gru
		from callback import LogMSE, LogVariable, NanStopping
		from keras.callbacks import EarlyStopping
		import keras.backend as K
		
		if self.model_type == 'rnn':
			if self.loss == 'yes': 		param, training_loss, evaluate_loglikelihood, callbacks, mse_pred, mae_pred, bias_init = get_loss_yes(self.x_train, self.wp_bp_train, batch_size = 4096)
			elif self.loss == 'clip': 	param, training_loss, evaluate_loglikelihood, callbacks, mse_pred, mae_pred, bias_init = get_loss_diff(self.x_train, self.wp_bp_train, batch_size = 4096)
			elif self.loss == 'mse': 	training_loss, bias_init = get_loss_mse(self.x_train, self.wp_bp_train, batch_size = 1024)
			elif self.loss == 'mse_sigma': training_loss, bias_init = get_loss_mse_sigma(self.x_train, self.wp_bp_train, batch_size = 1024)
			self.model = get_rnn(training_loss, self.x_train, bias_init, cell_type = self.gru_type, features_length = self.x_train.shape[2])
	
		min_val_loss, min_epoch = 10000000, 0
		print('x =', self.x_train.shape, 'y =', self.wp_bp_train.shape)
		 
		earlyStop = [EarlyStopping(monitor = 'val_loss', patience = 20, verbose = 2),]
		if self.loss == 'mse' or self.loss == 'mse_sigma':
			history = self.model.fit(self.x_train[:, self.interval:], self.wp_bp_train, epochs = self.epoch, batch_size = 1024, verbose = 2, 
			   	shuffle = False, validation_data = (self.x_valid[:, self.interval:], self.wp_bp_valid), callbacks = earlyStop)
		else:
			history = self.model.fit(self.x_train[:, self.interval:], self.wp_bp_train, epochs = self.epoch, batch_size = 1024, verbose = 2, 
			    shuffle = False, validation_data = (self.x_valid[:, self.interval:], self.wp_bp_valid), 
			    callbacks = callbacks + earlyStop + [LogVariable("name-param", param)])
		self.evaluation(self.x_train[:, self.interval:], self.x_valid[:, self.interval:])

		if self.rebuild > 0:
			self.x_train_rebuild = self.rebuild_one_cycle_rnn(self.x_train, self.is_under_train, self.x_train_origin, self.ts_train)
			self.x_valid_rebuild = self.rebuild_one_cycle_rnn(self.x_valid, self.is_under_valid, self.x_valid_origin, self.ts_valid)

			if self.rebuild == 1 or self.rebuild == 2:
				self.x_train = np.concatenate([self.x_train, self.x_train_rebuild], axis = 2)
				self.x_valid = np.concatenate([self.x_valid, self.x_valid_rebuild], axis = 2)
				print('Rebuild: x =', self.x_train.shape, 'y =', self.wp_bp_train.shape)
				print(self.x_train[0, 20:])
				if self.rebuild == 2: 
					self.x_train = self.pad_average(self.x_train, self.is_under_train)
					self.x_valid = self.pad_average(self.x_valid, self.is_under_valid)
				self.model = get_rnn(training_loss, self.x_train, bias_init, cell_type = self.gru_type, features_length = self.x_train.shape[2])
				
				if self.loss == 'mse' or self.loss == 'mse_sigma':
					history = self.model.fit(self.x_train[:, :], self.wp_bp_train, epochs = self.epoch, batch_size = 1024, verbose = 2, 
					   	shuffle = False, validation_data = (self.x_valid[:, :], self.wp_bp_valid), callbacks = earlyStop)
				else:
					history = self.model.fit(self.x_train[:, self.interval:], self.wp_bp_train, epochs = self.epoch, batch_size = 1024, verbose = 2, 
					    shuffle = False, validation_data = (self.x_valid[:, self.interval:], self.wp_bp_valid), 
					    callbacks = callbacks + earlyStop + [LogVariable("name-param", param)])

				self.evaluation(self.x_train[:, self.interval:], self.x_valid[:, self.interval:])
			elif self.rebuild == 3:
				fout = open(sys.argv[4], 'a')
				fout.write('rebuild\t')
				fout.close()

				self.x_train_rebuild = np.concatenate([np.copy(np.expand_dims(self.is_under_train[:, self.interval:], axis = 2)), self.x_train_rebuild[:, self.interval:]], axis = 2)
				self.x_valid_rebuild = np.concatenate([np.copy(np.expand_dims(self.is_under_valid[:, self.interval:], axis = 2)), self.x_valid_rebuild[:, self.interval:]], axis = 2)
				self.model = get_two_gru(training_loss, self.interval, cell_type = self.gru_type)
				self.x_train = [self.x_train, self.x_train_rebuild]
				self.x_valid = [self.x_valid, self.x_valid_rebuild]
				history = self.model.fit(self.x_train, self.wp_bp_train, epochs = self.epoch, batch_size = 1024, verbose = 2, 
					   	shuffle = False, validation_data = (self.x_valid, self.wp_bp_valid), callbacks = earlyStop)
				self.evaluation(self.x_train, self.x_valid)
		
	def evaluation(self, x_train, x_valid):
		pred_train = self.model.predict(x_train, batch_size = 1024, verbose = 2)
		pred_valid = self.model.predict(x_valid, batch_size = 1024, verbose = 2)
		
		if self.debug:
			print('real:  mean(train) = %.2f, mean(valid) = %.2f' %(np.mean(self.ts_train[:, -1]), np.mean(self.ts_valid[:, -1])))
			print('under: mean(train) = %.2f, mean(valid) = %.2f' %(np.mean(self.wp_bp_train[:, 1]), np.mean(self.wp_bp_valid[:, 1])))
			print('pred:  mean(train) = %.2f, mean(valid) = %.2f' %(np.mean(pred_train), np.mean(pred_valid)))
		
			pred_train = pred_train - np.mean(pred_train) + np.mean(self.ts_train[:, -1])
			utils.get_error(self.ts_train[:, -1], pred_train[:, 0], 'train')#, self.debug)
		
		pred_valid = pred_valid - np.mean(pred_valid) + np.mean(self.ts_valid[:, -1])
		utils.get_error(self.ts_valid[:, -1], pred_valid[:, 0], 'valid')#, self.debug)

		if self.debug:
			is_under_list = np.where(self.is_under_train[:, -1] == 1)[0]
			not_under_list = np.where(self.is_under_train[:, -1] == 0)[0]
			print('check len:', len(is_under_list), len(not_under_list), self.is_under_train.shape[0])
			pred_train = pred_train - np.mean(pred_train) + np.mean(self.ts_train[:, -1])
			utils.get_error(self.ts_train[is_under_list, -1], pred_train[is_under_list, 0], 'train (is) ')#, self.debug)
			utils.get_error(self.ts_train[not_under_list, -1], pred_train[not_under_list, 0], 'train (not)')#, self.debug)
			
			is_under_list = np.where(self.is_under_valid[:, -1] == 1)[0]
			not_under_list = np.where(self.is_under_valid[:, -1] == 0)[0]
			print('check len:', len(is_under_list), len(not_under_list), self.is_under_valid.shape[0])
			pred_valid = pred_valid - np.mean(pred_valid) + np.mean(self.ts_valid[:, -1])
			utils.get_error(self.ts_valid[is_under_list, -1], pred_valid[is_under_list, 0], 'valid (is) ')#, self.debug)
			utils.get_error(self.ts_valid[not_under_list, -1], pred_valid[not_under_list, 0], 'valid (not)')#, self.debug)
			
		
			print('train.MAE = ', end = '')
			for i in range(self.interval, self.interval*2):
				utils.MAE(self.ts_train[:, i], self.model.predict(self.x_train[:, i-self.interval:i], batch_size = 1024)[:, 0])
			print('\n           ', end = '')
			for i in range(self.interval, self.interval*2):
				print("%.1f" %(np.mean(self.model.predict(self.x_train[:, i-self.interval:i], batch_size = 1024)[:, 0]) - np.mean(self.ts_train[:, i])), end = "|")
			print('\nvalid.MAE = ', end = '')
			for i in range(self.interval, self.interval*2):
				utils.MAE(self.ts_valid[:, i], self.model.predict(self.x_valid[:, i-self.interval:i], batch_size = 1024)[:, 0])
			print('\n           ', end = '')
			for i in range(self.interval, self.interval*2):
				print("%.1f" %(np.mean(self.model.predict(self.x_valid[:, i-self.interval:i], batch_size = 1024)[:, 0]) - np.mean(self.ts_valid[:, i])), end = "|")
			print('\n')
			
			print('real =\t', end = '')
			for i in range(15):
				print('%.0f\t' %(self.ts_train[i, -1]), end='')
			print('\npred =\t', end = '')
			for i in range(15):
				print('%.0f\t' %(pred_train[i, 0]), end='')
			print('\nunder =\t', end = '')
			for i in range(15):
				print('%.0f\t' %(self.wp_bp_train[i, 1]), end='')
			print()

