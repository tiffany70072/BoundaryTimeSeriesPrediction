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

def under_sample_prob(ts, prob = 23, method = 'normal'):
	if prob == 77: 
		prob_OO = 0.839
		prob_XO = 0.161
		prob_OX = 0.524
		prob_XX = 0.476
		prob_O = 0.77
		prob_X = 0.23
	elif prob == 50:
		prob_OO = 0.5 * (0.476 + 0.839)
		prob_XO = 0.5 * (0.161 + 0.524)
		prob_OX = 0.5 * (0.161 + 0.524)
		prob_XX = 0.5 * (0.476 + 0.839)
		prob_O = 0.5
		prob_X = 0.5
	elif prob == 23:	# is_under: 1, True, O, 
		prob_OO = 0.476
		prob_XO = 0.524
		prob_OX = 0.161
		prob_XX = 0.839
		prob_O = 0.23
		prob_X = 0.77
	else:
		print('warning: no this prob value')
		exit()
	print("Under sample...", method)

	ts_under = np.copy(ts)
	is_under = np.zeros(ts.shape)
	for i in range(ts.shape[0]):
		dice = random.random()
		if dice < prob_O: # under sample now
			state_is_under = True
			ts_under[i][0] = sample_one_value(ts[i][0], method = method)
			is_under[i][0] = 1
		else: state_is_under = False
		
		for j in range(1, ts.shape[1]):
			dice = random.random()
			if state_is_under == True:
				if dice < prob_OO:
					ts_under[i][j] = sample_one_value(ts[i][j], method = method)
					is_under[i][j] = 1
				else: state_is_under = False
			else:
				if dice < prob_OX:
					ts_under[i][j] = sample_one_value(ts[i][j], method = method)
					state_is_under = True
					is_under[i][j] = 1
		#if i == 10: break
	#print('count_under =', count_under, "%.3f" %(count_under/float(count_under+count_not_under)))
	return ts_under, is_under

# prob(sample) = 0.5 * ts_percentage, value = halfnormal
def under_sample_outputs(ts, method = 'halfnormal'): 
	print('ts =', ts.shape)
	sample_all_value = list(np.random.randint(0, ts.shape[0], 5000))
	#all_value = np.reshape(np.copy(ts[:1000][-2]), [-1]) # don't calculate all value
	all_value = np.reshape(ts[sample_all_value][-2], [-1])
	print('all_value =', all_value.shape)
	sample_prob = np.zeros(ts.shape)
	from scipy.stats import percentileofscore
	#percentiles = [percentileofscore(all_value, i) for i in all_value]

	ts_under = np.copy(ts)
	is_under = np.zeros(ts.shape)
	for i in range(ts.shape[0]):
		for j in range(ts.shape[1]):
			dice = random.random()
			sample_prob_normalize = percentileofscore(all_value, ts[i][j]) * 0.01 * 0.5
			if dice < sample_prob_normalize:
				#print('dice =', dice, ', sample_prob =', sample_prob[i][j])
				is_under[i][j] = 1
				ts_under[i][j] = sample_one_value(ts[i][j], method = method)
		if i % 1000 == 999: print(i)
	return ts_under, is_under


def under_sample_pseudo_upper_bound(ts, method = 'halfnormal'):
	# reference:
	# X=A  (for simplicity), set B=cX+dY
	# lo(A, B) = c/[(c^2+d^2)^(0.5)]
	correlation = 0.7
	ts_under = np.copy(ts)
	is_under = np.zeros(ts.shape)
	for i in range(ts.shape[0]):
		noise = np.random.normal(0, scale = np.std(ts[i]), size = ts[i].shape)
		one_pseudo_ub = np.copy(ts[i]) + noise
		residue = np.median(ts[i]) - np.percentile(one_pseudo_ub, 25)
		one_pseudo_ub = one_pseudo_ub + residue

		for j in range(ts.shape[1]):
			if one_pseudo_ub[j] < ts[i][j]:
				#print('dice =', dice, ', sample_prob =', sample_prob[i][j])
				is_under[i][j] = 1
				ts_under[i][j] = one_pseudo_ub[j]
		if i % 1000 == 999: print(i)
	return ts_under, is_under

def under_sample_features(ts, feat, method='halfnormal'):
	feat_dim = feat.shape[2]
	random_alpha = np.random.uniform(0, 1, size = [feat_dim]) # importance of each features
	random_alpha = np.exp(random_alpha)/sum(np.exp(random_alpha)) # softmax
	print('random_alpha', random_alpha.shape, random_alpha)
	sample_prob = np.dot(feat, random_alpha)
	print('sample_prob =', sample_prob.shape, sample_prob[0][:10])
	sample_all_value = list(np.random.randint(0, feat.shape[0], 5000))
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



sample_method_list = ['prob_normal', 'prob_halfnormal', 'features', 'output', 'pseudo_upper_bound']
