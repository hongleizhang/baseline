#encoding:utf-8

import sys
try:
	import cPickle as pickle #python3归并为pickle
except:
	import pickle

import time
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.sparse as ss
from sklearn.cross_validation import train_test_split as tts
# from collections import defaultdict

from utils import log_to_cmd
"""
	数据处理相关类
	包括数据读取，保存
"""

#从txt文件中读取数据，返回DataFrame,Matrix,Array格式
def read_data(file_path, file_type='txt', sep='\t', header=None, return_format='DataFrame'):
	"""
	加载数据文件，返回指定类型 
	@@zhanghonglei

	Args:
		file_path: 数据文件存放路径
		file_type: 数据文件格式csv,dat,txt,json
		sep: 分隔字符串'\t',','
		header: 数据文件标题，0-表示从第1行开始，None-表示数据文件中无标题
		return_format: 返回的数据类型，返回DataFrame,Matrix,Array格式

	Returns:
		data: 返回指定格式的数据

	"""
	if file_type == 'csv' or file_type == 'dat':
		data=pd.read_csv(file_path, sep=sep, header=header)
	elif file_type == 'txt':
		data=pd.read_table(file_path, sep=sep, header=header)
	elif file_type =='json': #读取后缀名为.json的文件,默认返回dataframe格式；若不加lines=True的话，会报Trailing data的错误
		data=pd.read_json(file_path,lines=True)
	else:
		print('暂不支持此文件类型')

	if return_format == 'Matrix':
		data=np.mat(data)
	if return_format == 'Array':
		data=np.array(data)
	return data

#将DataFrame格式的数据转换为Rating Matrix 慢
def frame_to_mat(data,row_num,column_num):
	sparse_mat=ss.dok_matrix((row_num,column_num))
	for index,row in data.iterrows():
		user, item, rating = data.ix[index, 0], data.ix[index, 1], data.ix[index, 2]
		sparse_mat[int(user-1), int(item-1)]=int(rating)
	return sparse_mat
#ndarray
def frame_to_nmat(data,row_num,column_num):
	# sparse_mat=ss.dok_matrix((row_num,column_num))
	s=time.clock()

	mat=np.zeros((row_num,column_num))
	# user_item_mat=defaultdict(set)
	for index,row in data.iterrows():
		user, item, rating = data.ix[index, 0], data.ix[index, 1], data.ix[index, 2]
		mat[int(user-1)][int(item-1)]=int(rating)
	e=time.clock()
	log_to_cmd("array花费时间{}",e-s)
	return mat

#将DataFrame格式的数据转换为以user为键值和以item为键值的字典
def get_user_item_dict(data):
	user_items=defaultdict(set)
	item_users=defaultdict(set)
	for index,row in data.iterrows():
		user, item = data.ix[index, 0], data.ix[index, 1]
		user_items[user].add(item)
		item_users[item].add(user)
	return user_items,item_users

#将整个数据集分割为训练集与测试集
def train_test_split(data, train_size=0.7, test_size=0.3, random_state=0):
	data_train, data_test=tts(data,train_size=train_size, test_size=test_size, random_state=random_state)
	return pd.DataFrame(data_train), pd.DataFrame(data_test)

#保存内存对象数据到磁盘(python原生对象)
def save_pickle_to_disk(filename, value):
	pickle.dump(value,open(filename, 'wb'))
	pass

#从磁盘文件中读取数据到内存对象(python原生对象)
def read_pickle_from_disk(filename):
	data=pickle.load(open(filename, 'rb'))
	return data

#保存内存对象数据到磁盘(python扩展对象-dok_matrix)
def save_data_to_disk(filename, value):
	sio.mmwrite(filename,value)
	pass

#从磁盘文件中读取数据到内存对象(python扩展对象-dok_matrix)
def read_data_from_disk(filename):
	data=sio.mmread(filename)
	return data

def write_data(value,filename='default.csv'):
	data=pd.DataFrame(value)
	columns=list(data.columns)
	print(columns)
	data.to_csv(filename,columns=columns,index=False,index_label=False)