# Jan/12
import numpy as np 
from numpy.random import seed
seed(1)
from random import seed
seed(1)
import random

def sample_one_value(value, min_bound = 0, max_bound = 1, method = 'normal'):
	if method == 'uniform': new_value = random.random()*value
	if method == 'normal': new_value = value*random.gauss(0.5, 0.2)
	if method == 'halfnormal':
		ratio = random.gauss(1, 0.4)
		if ratio > 1: ratio = 2 - ratio
		new_value = value*ratio
	new_value = max(0, min([value, new_value]))
	return round(new_value, 0)

def under_sample_outputs(ts, method = 'normal'): 
	print('ts =', ts.shape)
	sample_all_value = list(np.random.randint(0, ts.shape[0], 5000))
	#all_value = np.reshape(np.copy(ts[:1000][-2]), [-1]) # don't calculate all value
	all_value = ts[sample_all_value]
	print('all_value =', all_value.shape)
	all_value = all_value[~np.isnan(all_value)]
	print('all_value (remove nan) =', all_value.shape)
	sample_prob = np.zeros(ts.shape)
	from scipy.stats import percentileofscore
	#percentiles = [percentileofscore(all_value, i) for i in all_value]

	ts_under = np.copy(ts)
	is_under = np.zeros(ts.shape)
	for i in range(ts.shape[0]):
		dice = random.random()
		sample_prob_normalize = percentileofscore(all_value, ts[i]) * 0.01 * 0.5
		if dice < sample_prob_normalize:
			is_under[i] = 1
			ts_under[i] = sample_one_value(ts[i], method = method)
		if i % 5000 == 4999: print(i+1),
	print()
	return ts_under

def under_sample_pseudo_upper_bound(ts, prob, method = 'normal'):
	count = 0
	ts_under = []

	for i in range(np.sum(np.isnan(ts))):
		ts_one = []
		while True:
			if count == len(ts) or np.isnan(ts[count]) == 1: 
				count += 1
				break
			ts_one.append(ts[count])
			count += 1
		ts_under_one = list(np.copy(ts_one))
		
		if len(ts_one) > 0:
			noise = np.random.normal(0, scale = np.std(ts_one), size = ts.shape)
			one_pseudo_ub = np.copy(ts) + noise
			if prob == 25: 
				residue = np.mean(ts_one) #- np.percentile(all_value, 25) # 25%
				#residue = np.percentile(ts_one, 75) - np.percentile(ts_one, 25)
			elif prob == 50: 
				#residue = 0.5 * (np.percentile(ts_one, 75) - np.percentile(ts_one, 25)) # 50%
				residue = 0.5 * np.mean(ts_one)
			one_pseudo_ub = one_pseudo_ub + residue

			for j in range(len(ts_one)):
				if one_pseudo_ub[j] < ts_one[j]: ts_under_one[j] = one_pseudo_ub[j]
				if j % 5000 == 4999: print(j+1),
			ts_under += ts_under_one
			ts_under.append('nan')
		else: 
			for i in range(len(ts_one)+1): ts_under.append('nan')
			i -= 1
	
	print('ts =', ts.shape, ts[:30])
	print('ts_under =', len(ts_under), ts_under[:30])
	ts_under = np.array(ts_under, dtype = np.float)
	#print(np.nanmean(ts_under))
	ts_under = np.clip(ts_under, 0, 100000)
	
	import pdb
	#pdb.set_trace()
	assert ts_under.shape == ts.shape, 'error in under sample ub !'
	return ts_under

'''
def under_sample_features(ts, feat, method='halfnormal'):
	feat_dim = feat.shape[2]
	random_alpha = np.random.uniform(0, 1, size = [feat_dim]) # importance of each features
	random_alpha = np.exp(random_alpha)/sum(np.exp(random_alpha)) # softmax
	print('random_alpha', random_alpha.shape, random_alpha)
	sample_prob = np.dot(feat, random_alpha)
	print('sample_prob =', sample_prob.shape, sample_prob[0][:10])
	sample_all_value = list(np.random.randint(0, feat.shape[0], 5000))
	#all_value = np.reshape(np.copy(sample_prob[:1000][-2]), [-1])
	all_value = np.reshape(np.copy(sample_prob[sample_all_value][-2]), [-1])

	from scipy.stats import percentileofscore
	
	ts_under = np.copy(ts)
	is_under = np.zeros(ts.shape)
	for i in range(ts.shape[0]):
		for j in range(ts.shape[1]):
			dice = random.random()
			sample_prob_normalize = percentileofscore(all_value, sample_prob[i][j]) * 0.005
			if dice < sample_prob_normalize:
				is_under[i][j] = 1
				ts_under[i][j] = sample_one_value(ts[i][j], method = method)
		if i % 1000 == 999: print(i)
	return ts_under, is_under
'''


sample_method_list = ['prob_normal', 'prob_halfnormal', 'features', 'output', 'pseudo_upper_bound']
