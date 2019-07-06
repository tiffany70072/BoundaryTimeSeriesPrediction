# Jan/12

import numpy as np 
from os import path as Path

def get_is_under(ts, ts_under):
	is_under = (ts != ts_under)
	print('ratio =', np.mean(is_under))
	return is_under

def load_pm25_data(name='prob77'):
	if name == 'prob77': path = '../src_1017_underestimation/data_prob77_1211/'
	elif name == 'prob50': path = '../src_1017_underestimation/data_prob50_1229/'
	elif name == 'output': path = '../src_1017_underestimation/data_output_1211/'
	elif name == 'halfnormal': path = '../src_1017_underestimation/data_prob_halfnormal_1220/'
	print('Load pm25 data')
	ts_train = np.load(path + 'ts_train.npy')
	ts_valid = np.load(path + 'ts_valid.npy')
	ts_under_train = np.load(path + 'ts_under_train.npy')
	ts_under_valid = np.load(path + 'ts_under_valid.npy')
	is_under_train = np.load(path + 'is_under_train.npy')
	is_under_valid = np.load(path + 'is_under_valid.npy')
	print('ts =', ts_train.shape, ts_valid.shape)
	print('ts_under =', ts_under_train.shape, ts_under_valid.shape)
	return ts_train, ts_valid, ts_under_train, ts_under_valid, is_under_train, is_under_valid

def load_default_data(with_features = False):
	path = '../src_1017_underestimation/'
	print('Load train valid data')
	ts_train = np.load(path + 'data/ts_train_1113.npy')
	ts_valid = np.load(path + 'data/ts_valid_1113.npy')
	ts_under_train = np.load(path + 'data/ts_under_train_1113.npy')
	ts_under_valid = np.load(path + 'data/ts_under_valid_1113.npy')
	is_under_train = np.load(path + 'data/is_under_train_1113.npy')
	is_under_valid = np.load(path + 'data/is_under_valid_1113.npy')
	return ts_train, ts_valid, ts_under_train, ts_under_valid, is_under_train, is_under_valid

def load_data(with_features = False):
	print('Load train valid data')
	ts_train = np.load('data/ts_train_1113.npy')
	ts_valid = np.load('data/ts_valid_1113.npy')
	ts_under_train = np.load('data/ts_under_train_1113.npy')
	ts_under_valid = np.load('data/ts_under_valid_1113.npy')
	is_under_train = np.load('data/is_under_train_1113.npy')
	is_under_valid = np.load('data/is_under_valid_1113.npy')
	ts_features_train = np.load('data/ts_features_train_1113.npy')
	ts_features_valid = np.load('data/ts_features_valid_1113.npy')

	if with_features == False:
		return ts_train, ts_valid, ts_under_train, ts_under_valid, is_under_train, is_under_valid
	else:
		return ts_train, ts_valid, ts_under_train, ts_under_valid, is_under_train, is_under_valid, ts_features_train, ts_features_valid

def load_pm25_uniform_data(name='prob50'):
	path = '../src_1017_underestimation/data_prob50_1229/'
	print('Load pm25 uniform data')
	ts_train = np.load(path + 'ts_train.npy')
	ts_valid = np.load(path + 'ts_valid.npy')

	if name == 'prob23': path = '../src_1017_underestimation/data_pm2.5_prob/pm2.5_prob_23_uniform/'
	elif name == 'prob50': path = '../src_1017_underestimation/data_pm2.5_prob/pm2.5_prob_50_uniform/'
	elif name == 'prob77': path = '../src_1017_underestimation/data_pm2.5_prob/pm2.5_prob_77_uniform/'
	ts_under_train = np.load(path + 'ts_under_train.npy')
	ts_under_valid = np.load(path + 'ts_under_valid.npy')
	if Path.exists(path + 'is_under_train.npy'):
		is_under_train = np.load(path + 'is_under_train.npy')
		is_under_valid = np.load(path + 'is_under_valid.npy')
	else:
		is_under_train = get_is_under(ts_train, ts_under_train)
		is_under_valid = get_is_under(ts_valid, ts_under_valid)
	print('ts =', ts_train.shape, ts_valid.shape)
	print('ts_under =', ts_under_train.shape, ts_under_valid.shape)
	return ts_train, ts_valid, ts_under_train, ts_under_valid, is_under_train, is_under_valid

def load_O3_data(name='prob77'):
	path = '../src_1017_underestimation/data_O3/'
	print('Load O3 data')
	ts_train = np.load(path + 'ts_train.npy')
	ts_valid = np.load(path + 'ts_valid.npy')

	if name == 'prob50': path = '../src_1017_underestimation/data_O3/O3_prob50_normal_0112/'
	ts_under_train = np.load(path + 'ts_under_train.npy')
	ts_under_valid = np.load(path + 'ts_under_valid.npy')
	is_under_train = np.load(path + 'is_under_train.npy')
	is_under_valid = np.load(path + 'is_under_valid.npy')
	print('ts =', ts_train.shape, ts_valid.shape)
	print('ts_under =', ts_under_train.shape, ts_under_valid.shape)
	return ts_train, ts_valid, ts_under_train, ts_under_valid, is_under_train, is_under_valid

def load_script_data():
	import sys
	if sys.argv[1] == 'default':
		return load_default_data()
	elif sys.argv[1] == 'prob77':
		return load_pm25_data('prob77')
	elif sys.argv[1] == 'prob50':
		return load_pm25_data('prob50')
	elif sys.argv[1] == 'output':
		return load_pm25_data('output')
	elif sys.argv[1] == 'prob50_uniform':
		return load_pm25_uniform_data('prob50')	
	elif sys.argv[1] == 'O3_prob50':
		return load_O3_data('prob50')

def load_KDD_2018_data(place = 'ld'):
	import sys
	if place == 'ld':   folder = 'data_pm25_2018/ts_under_sample_ld/'
	elif place == 'bj': folder = 'data_pm25_2018/ts_under_sample_bj/'
	elif place == 'parking': folder = 'data_parking/'
	elif place == 'taxi': folder = 'ts_under_taxi/'
	
	if sys.argv[1] == '23': filename = '23.npy'
	elif sys.argv[1] == '50': filename = '50.npy'
	elif sys.argv[1] == '77': filename = '77.npy'
	elif sys.argv[1] == '25': filename = '25.npy'
	elif sys.argv[1] == '75': filename = '75.npy'
	elif sys.argv[1] == 'half': filename = '50_half.npy'
	elif sys.argv[1] == 'uniform': filename = '50_uniform.npy'
	elif sys.argv[1] == 'output': filename = '25_output.npy'
	elif sys.argv[1] == 'ub': filename = '25_ub.npy'
	elif sys.argv[1] == 'lof': filename = 'lof.npy'
	elif sys.argv[1] == 'o3': filename = '50_o3.npy'

	if sys.argv[1] != 'o3':
		ts_train = np.load(folder + 'ts_train.npy')
		ts_valid = np.load(folder + 'ts_valid.npy')
	else: 
		ts_train = np.load(folder + 'ts_train_o3.npy')
		ts_valid = np.load(folder + 'ts_valid_o3.npy')
	ts_under_train = np.load(folder + 'ts_train_' + filename)
	ts_under_valid = np.load(folder + 'ts_valid_' + filename)
	is_under_train = ~(ts_train == ts_under_train)
	is_under_valid = ~(ts_valid == ts_under_valid)
	print('shape =', ts_train.shape, ts_valid.shape, ts_under_train.shape, ts_under_valid.shape)

	# check under ratio
	'''print('\ntrain')
	print('count_under_ratio = %.3f' %(np.sum(np.sum(is_under_train))/float(is_under_train.shape[0]*is_under_train.shape[1])), end = ', ')
	print(np.sum(np.sum(is_under_train)), is_under_train.shape[0])
	print('under_value_ratio(overall) = %.3f' %(np.nanmean(ts_under_train)/np.nanmean(ts_train)))
	print('under_value_ratio(only) = %.3f' %(np.nanmean(ts_under_train*is_under_train)/np.nanmean(ts_train*is_under_train)))
	print('\nvalid')
	print('count_under_ratio = %.3f' %(np.sum(np.sum(is_under_valid))/float(is_under_valid.shape[0]*is_under_valid.shape[1])), end = ', ')
	print(np.sum(np.sum(is_under_valid)), is_under_valid.shape[0])
	print('under_value_ratio(overall) = %.3f' %(np.nanmean(ts_under_valid)/np.nanmean(ts_valid)))
	print('under_value_ratio(only) = %.3f' %(np.nanmean(ts_under_valid*is_under_valid)/np.nanmean(ts_valid*is_under_valid)))
	'''
	return ts_train, ts_valid, ts_under_train, ts_under_valid, is_under_train, is_under_valid

def main():
	load_KDD_2018_data()

if __name__ == '__main__':
	main()
	