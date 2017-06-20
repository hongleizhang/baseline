#encoding:utf-8

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils as util
import data_processing as dp
import matrix_factorization as mf
import scipy.sparse as ss
import metrics
import similarity as si
try:
	reload(util)
	reload(dp)
	reload(mf)
	reload(metrics)
except: #适用于python3
	import imp
	imp.reload(util)
	imp.reload(dp)
	imp.reload(mf)
	imp.reload(metrics)

DATA_RATING_PATH='../../dataset/ciao/ciao_rating.txt'
DATA_TRUST_PATH='../../dataset/ciao/ciao_trust.txt'
DATA_RATING_PATH_ML='../../dataset/ml-1m/ratings.dat'


USER_NUM=7375 #7375 6040
ITEM_NUM=106797 # 3952
m=None
bu=None
bi=None
p=None
q=None




#测试最相似用户
# 返回指定用户k个最近邻
def imgNearestMat(user_item_mat,userID,k=10,sim=si.cosine_distance):
	knn={}
	scorelist={}
	start=time.clock()
	targetId=imgID-1
	# for targetId in range(imgID,imgID+1):
	for temp in range(7375):
		if targetId==temp: continue #如果和自己比较，则跳过
		knn.setdefault(targetId,{})
		simscore=sim(user_item_mat[targetId],user_item_mat[temp])
		scorelist.setdefault(temp,0)
		scorelist[temp]=simscore
	print 'the first time is %s' % (time.clock()-start)
	newscorelist=copy.copy(scorelist)
	newscorelist=sorted(newscorelist.items(),key=lambda asd:asd[1],reverse=True)[:k]  #python3没有iteritems
	end=time.clock()
	print (end-start)
	print newscorelist
	return newscorelist

def main():

	global m,bu,bi,p,q
	start=time.clock()
	util.log_to_cmd("...biasMF...")
	util.log_to_cmd("读取数据...")
	train_data_mat=ss.dok_matrix(dp.read_data_from_disk("train_data_mat.mtx"))
	test_data_mat=ss.dok_matrix(dp.read_data_from_disk("test_data_mat.mtx"))
	# data_rating_f=dp.read_data(DATA_RATING_PATH)
	# util.log_to_cmd("数据分为训练集与测试集...")
	# train_data, test_data=dp.train_test_split(data_rating_f)
	# util.log_to_cmd("转化数据为稀疏矩阵...")
	# train_data_mat=dp.frame_to_mat(train_data,USER_NUM,ITEM_NUM)
	# test_data_mat=dp.frame_to_mat(test_data,USER_NUM,ITEM_NUM)
	# util.log_to_cmd("数据持久化到磁盘...")
	# dp.save_data_to_disk('train_data_mat',train_data_mat)
	# dp.save_data_to_disk('test_data_mat',test_data_mat)
	end=time.clock()
	util.log_to_cmd("读取数据完毕",None,end-start)

	util.log_to_cmd("矩阵分解中...")
	m,bu,bi,p,q=mf.matrix_factorization(train_data_mat,45)
	util.log_to_cmd("MSE分析中...")
	mse=metrics.mf_mse(test_data_mat,p,q,m,bu,bi)
	util.log_to_cmd("mse值：{}",mse)

def main_base():
	util.log_to_cmd("...baseMF...")
	util.log_to_cmd("读取数据...")
	train_data_mat=ss.dok_matrix(dp.read_data_from_disk("train_data_mat.mtx"))
	test_data_mat=ss.dok_matrix(dp.read_data_from_disk("test_data_mat.mtx"))
	pp,qq=mf.matrix_factorization_base(train_data_mat,45)
	util.log_to_cmd("MSE分析中...")
	mse=metrics.mf_mse(test_data_mat,pp,qq)
	util.log_to_cmd("mse值：{}",mse)
	pass
	
if __name__=="__main__":
	main()