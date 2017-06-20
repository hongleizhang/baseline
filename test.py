#encoding:utf-8

import os
import numpy as np
import pandas as pd

try:
	import matplotlib.pyplot as plt
	import imp
except:
	print('【Error】:This implementation requires the matplotlibs module')
	os._exit(1)

import data_processing as dp
imp.reload(dp)

def main():
	pass


def test():
	for i in range(10):
		if i == 5:
			break
		print(i)
	else:
		print('遇到5了吧')


def write_json_to_csv(xuhao=0):
	file_path='../../dataset/yelp_round9/'
	file_out_path='../../dataset/yelp_round9/csv/'
	file_list=['yelp_academic_dataset_business','yelp_academic_dataset_checkin','yelp_academic_dataset_review','yelp_academic_dataset_tip','yelp_academic_dataset_user']
	file=file_list[xuhao]
	file_name=file+'.json'
	data=dp.read_data(file_path+file_name,'json')
	dp.write_data(data,file_out_path+file+'.csv')

if __name__=="__main__":
	test()