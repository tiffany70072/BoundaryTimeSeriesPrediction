import numpy as np 
import sys

# ER, SMAPE, RMLSE, RMSE (MAE, MSE, MAPE, WMAE)
def RMSE(real, pred):
	error_list = []
	for i in range(0, real.shape[0], 10000):
		error_list.append(np.mean(np.square(real[i:i+10000] - pred[i:i+10000])))
	
	rmse = np.sqrt(np.mean(error_list))
	print("%.2f" %rmse, end = "|")
	return rmse

def RMLSE(real, pred):
	error_list = []
	for i in range(0, real.shape[0], 10000):
		error_list.append(np.mean(np.square(np.log(1+real[i:i+10000]) - np.log(1+pred[i:i+10000]))))
	
	rmlse = np.sqrt(np.mean(error_list))
	print("%.2f" %rmlse, end = "|")
	return rmlse

def SMAPE(real, pred):
	error_list = []
	for i in range(0, real.shape[0], 10000):
		error_list.append(np.mean(np.abs(real[i:i+10000] - pred[i:i+10000])/(np.abs(real[i:i+10000]) + np.abs(pred[i:i+10000]))))
	
	smape = (200 * np.mean(error_list))
	print("%.2f" %smape, end = "|")
	return smape

def MAE(real, pred):
	error_list = []
	for i in range(0, real.shape[0], 10000):
		error_list.append(np.mean(np.abs(real[i:i+10000] - pred[i:i+10000])))
	
	mae = np.mean(error_list)
	print("%.2f" %mae, end = "|")
	return mae

def get_error(real, pred, name):
	print(name, end = "|")
	mae   = MAE(real, pred)
	rmse  = RMSE(real, pred)
	smape = SMAPE(real, pred)
	rmlse = RMLSE(real, pred)
	
	if len(sys.argv) > 4: 
		#print('write file')
		fout = open(sys.argv[4], 'a')
		for error in [mae, rmse, smape, rmlse]: 
			fout.write("%.2f\t" %error)
		fout.write('\n')
		fout.close()
	print()

def repeat_front(ts_under, is_under): # 把往前數第一個沒有低估的，複製拿過來捕
	ts_new = np.copy(ts_under)
	for i in range(ts_under.shape[0]):
		for j in range(ts_under.shape[1]-1,-1,-1):
			if is_under[i][j] == 0: continue
			for k in range(j-1, -1, -1):
				if is_under[i][k] == 0:
					ts_new[i][j] = np.copy(ts_under[i][k])
					break
	return ts_new

def concat_front(ts_under, is_under, interval): # 把有低估的去掉，整體長度變短
	new_ts = np.empty([ts_under.shape[0], interval])
	for i in range(new_ts.shape[0]):
		count = interval - 1
		for j in range(ts_under.shape[1]-1, -1, -1):
			if is_under[i][j] == 0:
				new_ts[i][count] = ts_under[i][j]
				count -= 1
			if count == -1: break
	return new_ts

def remove_first(ts_under, is_under, interval):
	new_ts = np.copy(ts_under[:, 0:1]) # only update the first one
	count_error = 0
	for i in range(ts_under.shape[0]):
		if is_under[i, 0] == 1:
			for j in range(1, interval):
				if is_under[i, j] == 0:
					new_ts[i, 0] = ts_under[i, j]
			if np.sum(is_under[i, :]) == interval: count_error += 1	
	print('remove_first, error =', count_error, count_error/float(new_ts.shape[0]))
	new_ts = np.concatenate([new_ts, ts_under[:, 1:]], axis = 1)
	return new_ts
	
def remove_mix(ts_under, is_under, interval): # only update when id = 11 is under and id = 10, 9 is not under
	assert interval == 24

	ts_new = np.where(is_under, ts_under*2, ts_under)
	count_error = 0
	for i in range(ts_under.shape[0]):
		for j in range(4, 18):
			if is_under[i, j] == 1:
				if is_under[i, j-1] == 0: ts_new[i][j] = (ts_under[i][j-1] + ts_new[i][j]) * 0.5
				elif is_under[i, j+1] == 0: ts_new[i][j] = (ts_under[i][j+1] + ts_new[i][j]) * 0.5
				elif is_under[i, j-2] == 0: ts_new[i][j] = (ts_under[i][j-2] + ts_new[i][j]) * 0.5
				elif is_under[i, j+2] == 0: ts_new[i][j] = (ts_under[i][j+2] + ts_new[i][j]) * 0.5
				elif is_under[i, j-3] == 0: ts_new[i][j] = (ts_under[i][j-3] + ts_new[i][j]) * 0.5
				elif is_under[i, j+3] == 0: ts_new[i][j] = (ts_under[i][j+3] + ts_new[i][j]) * 0.5
				elif is_under[i, j-4] == 0: ts_new[i][j] = (ts_under[i][j-4] + ts_new[i][j]) * 0.5
				elif is_under[i, j+4] == 0: ts_new[i][j] = (ts_under[i][j+4] + ts_new[i][j]) * 0.5
				else: 
					count_error += 1	
				
	#print('debug, check middle =', np.mean(ts_under), np.mean(ts_new))
	#print('debug =', ts_under[:3])
	#print('debug =', ts_new[:3])
	#print('debug =', is_under[:3])
	#print('remove index 11, error =', count_error, count_error/float(ts_new.shape[0]*(18-4)))
	return np.copy(np.expand_dims(ts_new, axis = 2))
	