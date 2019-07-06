# Create at March/5/2019
# Dual stage rebuild method for time series prediction
# usage: time CUDA_VISIBLE_DEVICES=2 python3 main_two_rebuild.py 

from underEstimation import UnderEstimation

if __name__ == '__main__':
	underEst = UnderEstimation()
	underEst.train()
	print(underEst.message)