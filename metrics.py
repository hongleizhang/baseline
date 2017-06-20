#encoding:utf-8

import numpy as np
import pandas as pd
from sklearn import metrics
import math

#针对于Top-N推荐问题主流度量指标

#准确率
def precision_score(y_true, y_pred):
	score=metrics.precision_score(y_true, y_pred)
	return score

#召回率
def recall_score(y_true, y_pred):
	score=metrics.recall_score(y_true, y_pred)
	return score

#F1值
def f1_score(y_true, y_pred):
	score=metrics.f1_score(y_true, y_pred)
	return score

#针对于【评分预测】推荐问题主流度量指标

#绝对平均误差
def mean_absolute_error(y_true, y_pred):
	score=metrics.mean_absolute_error(y_true, y_pred)
	return score

#均方误差
def mean_squared_error(y_true, y_pred):
	score=metrics.mean_squared_error(y_true, y_pred)
	return score

def mf_mse(test_train_mat,p,q,m=None,bu=None,bi=None):

	data_nonzero=test_train_mat.nonzero()
	data_zip=list(zip(data_nonzero[0],data_nonzero[1]))
	item_count=len(data_zip)
	s=0
	for item in range(item_count):
		pr=0
		x, y=data_zip[item]
		tr=test_train_mat[x,y]
		if m==None:
			pr=np.dot(p[x,:],q[:,y])
		else:
			pr=m+bu[x]+bi[y]+np.dot(p[x,:],q[:,y])
		s+=pow((tr-pr),2)
	s/=item_count
	return math.sqrt(s)