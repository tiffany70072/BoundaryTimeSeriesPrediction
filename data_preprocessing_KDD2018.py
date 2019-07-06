'''
Jan/12
Mode 1: 
	Given a 1-D array, with splited train/valid data (outputs, pseudo)
	Step 1: under_sample
	Step 2: cut to window size
	Step 3: save data
Mode 2:
	Given a 2-D array, with (None, window size)
	Step 1: check size
	Step 2: return is_under matrix
'''

import numpy as np 
from numpy.random import seed
seed(1)
from sklearn.model_selection import train_test_split
from pathlib import Path
#import os.path
from os import path
import sys

import under_sample_uncut as under_sample

def output_ts(ts, name):
	print(name, end = ' -- ')
	for i in range(50): print('%.1f' %ts[i], end = ' ')
	print("")

class DataPreprocessor(object):
	def __init__(self):
		# data variables
		#self.path = 'data_pm25_2018/ts_under_sample_bj/'
		self.path = 'data_pm25_2018/ts_under_sample_ld/'

		# under sample variables
		select_is_under_list = ['prob', 'outputs', 'pseudo']
		select_value_list = ['normal', 'uniform', 'halfnormal']
		self.select_is_under = select_is_under_list[int(sys.argv[1])] # 1 and 2 for me
		self.select_value = select_value_list[0]

		if int(sys.argv[1]) == 1: name = '_25_output'
		if int(sys.argv[1]) == 2: name = '_25_ub'
		if int(sys.argv[1]) == 3: name = '_50_ub'
		
		# main
		for split in ['valid', 'train']:
			self.get_raw_data(split)
			self.ts_under, self.is_under = self.get_under_sample(self.ts)
			self.ts = self.cut_data(self.ts)
			self.ts_under = self.cut_data(self.ts_under)
			self.remove_nan()
			self.save_ts_file(split)
			self.save_file(name, split)

	def get_raw_data(self, split):
		filename = self.path + 'ts_' + split + '_uncut.npy' 
		#filename = self.path + 'ts_under_sample_bj/ts_' + split + '_uncut.npy'

		if path.exists(filename):
			print('already train test split')
			self.ts = np.load(filename, fix_imports = 1, encoding = 'latin1')
			print('nan ratio =', np.mean(np.isnan(self.ts)))
			print('ts =', self.ts[1000:1010])
			self.ts = np.clip(self.ts, 0, 100000)
			if np.isnan(self.ts[0]) == True: 
				self.ts[0:-1] = self.ts[1:]
				self.ts[-1] = 'nan'
		else:
			print('No this path: ', filename)
			exit()
	'''
	def own_train_test_split(self):
		print('train test split')
		self.ts_train, self.ts_valid = train_test_split(self.ts, test_size = 0.25, random_state = 17)
		np.save(self.path + 'ts_train.npy', self.ts_train)
		np.save(self.path + 'ts_valid.npy', self.ts_valid)
	'''

	def save_ts_file(self, split):
		filepath = self.path + 'ts_' + split + '.npy'
		print('filepath =', filepath)
		#if not Path(filepath).is_dir(): os.mkdir(filepath)#, 777)
		print('folder is created')

		np.save(filepath, self.ts)
		print(split + '.shape =', self.ts.shape, self.is_under.shape)
		print('saved successfully')

	def save_file(self, name, split):
		# create folder
		filepath = self.path + 'ts_' + split + name
		print('filepath =', filepath)
		np.save(filepath, self.ts_under)
		print(split + '.shape =', self.ts_under.shape, self.is_under.shape)
		print('saved successfully')

	def get_under_sample(self, ts):
		'''Available Methods:
		select_is_under
			1. Prob (with 23%, 50%, 77%)
			2. Output
			3. Pseudo upper bound
		select_value (method)
			1. Normal (normal)
			2. Half-normal (halfnormal)
			3. Uniform (uniform)
		'''
		
		if self.select_is_under == 'prob':      ts_under = under_sample.under_sample_prob(ts, method = self.select_value, prob = 50)
		elif self.select_is_under == 'outputs': ts_under = under_sample.under_sample_outputs(ts, method = self.select_value)
		elif sys.argv[1] == '2':  				ts_under = under_sample.under_sample_pseudo_upper_bound(ts, 25, method = self.select_value)
		elif sys.argv[1] == '3':  				ts_under = under_sample.under_sample_pseudo_upper_bound(ts, 50, method = self.select_value)
		elif self.select_is_under == 'pseudo':  ts_under = under_sample.under_sample_pseudo_upper_bound(ts, prob = 25, method = self.select_value)
		
		is_under = (ts != ts_under)
		output_ts(ts, 'ts      ')
		output_ts(ts_under, 'ts_under')
		output_ts(is_under, 'is_under')
		print('count_under_ratio = %.4f' %(np.sum(np.sum(is_under))/float(is_under.shape[0])), end = ', ')
		print(np.sum(np.sum(is_under)), is_under.shape[0])
		print('under_value_ratio(overall) = %.4f' %(np.nanmean(ts_under)/np.nanmean(ts)))
		print('under_value_ratio(only) = %.4f' %(np.nanmean(ts_under*is_under)/np.nanmean(ts*is_under)))

		return ts_under, is_under

	def cut_data(self, ts, window_size = 49): # return cut 2D array
		ts_2D = np.zeros([ts.shape[0] - window_size + 1, window_size])
		for i in range(ts.shape[0] - window_size + 1):
			ts_2D[i] = np.copy(ts[i:i+window_size])
		print('ts_2D =', ts_2D.shape)
		return ts_2D

	def remove_nan(self):
		no_nan_id = []
		for i in range(self.ts.shape[0]):
			if np.sum(np.isnan(self.ts[i])) == 0: no_nan_id.append(i)
		print('no nan length =', len(no_nan_id))

		self.ts_under = np.copy(self.ts_under[no_nan_id])
		self.is_under = np.copy(self.is_under[no_nan_id])
		self.ts = np.copy(self.ts[no_nan_id])

if __name__ == '__main__':
	dataPreprocessor = DataPreprocessor()

