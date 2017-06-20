#encoding:utf-8

import time
import numpy as np
import pandas as pd
import scipy.sparse as ss
import math

import utils as util

def initP(row_num, f):
	P=ss.dok_matrix((row_num,f))
	for i in range(row_num):
		P[i,:]=np.random.rand(1,f)/math.sqrt(f)
	return P

#慢
def initQ(column_num, f):
	Q=ss.dok_matrix((column_num,f))
	for j in range(column_num):
		Q[j,:]=np.random.rand(1,f)/math.sqrt(f)
	return Q.T

def initNP(row_num, f):
	P=np.zeros((row_num,f))
	for i in range(row_num):
		P[i,:]=np.random.rand(1,f)/math.sqrt(f)
	return P

def initNQ(column_num, f):
	Q=np.zeros((column_num,f))
	for j in range(column_num):
		Q[j,:]=np.random.rand(1,f)/math.sqrt(f)
	return Q.T

def initBu(row_num):
	bu=np.zeros((row_num))
	return bu

def initBi(column_num):
	bi=np.zeros((column_num))
	return bi

def matrix_factorization(data, f, times=500, t=1e-4, alpha=0.02, beta=0.02 ): #alpha=0.0002
	"""
	矩阵分解

	Args:
		data: 需要进行分解的矩阵R
		f: 隐含因子的个数
		e: 误差的预制，默认为0.001
		times: 最大迭代次数
		alpha: 学习率
		beta: 正则化系数
	Returns:
		返回P,Q矩阵

	"""
	util.log_to_cmd("开始运算")
	row_num, column_num=data.shape
	P=initNP(row_num, f) #用户潜在特征矩阵
	util.log_to_cmd("初始化P完毕")
	Q=initNQ(column_num, f) #项目潜在特征矩阵
	util.log_to_cmd("初始化Q完毕")
	bu=initBu(row_num) #用户偏置向量
	util.log_to_cmd("初始化bu完毕")
	bi=initBi(column_num) #项目偏置向量
	util.log_to_cmd("初始化bi完毕")
	m=data.sum()/len(data.items()) #评分均值 m=4.1611963852267015
	util.log_to_cmd("平均分计算完毕{}",m)
	data_nonzero=data.nonzero()
	data_zip=list(zip(data_nonzero[0],data_nonzero[1])) #zip 在py2中返回list,在py3中返回迭代器
	util.log_to_cmd("开始迭代")
	eb=0
	for current_time in range(times):
		util.log_to_cmd("第{}次迭代",current_time)
		e=0
		# ss=time.clock()
		for item in range(len(data_zip)): #空读0.02 稀疏矩阵计算0.03603109688992845 数组计算0.0010
			x, y=data_zip[item]
			r=data[x,y]
			dot=np.dot(P[x,:],Q[:,y])
			#计算预测评分误差
			err=r-(m+bu[x]+bi[y]+dot)
			e+=pow(err,2)

			#更新偏置项
			bu[x]+=alpha*(err-beta*bu[x])
			bi[y]+=alpha*(err-beta*bi[y])
			#更新P,Q
			for i in range(f):  
				P[x,i]+=alpha*(err*Q[i,y]-beta*P[x,i])
				Q[i,y]+=alpha*(err*P[x,i]-beta*Q[i,y])
				#计算损失函数
				e+=(beta/2)*(pow(P[x,i],2)+pow(Q[i,y],2))
		util.log_to_cmd("损失值e={}",e)
		alpha*=0.9
		# ee=time.clock()
		# util.log_to_cmd("一趟遍历时间为:{}",ee-ss)
       	#设置阈值停止迭代
		if np.abs(e-eb)<t:
			util.log_to_cmd("到达阈值终止条件{}",e)
			break
		eb=e #更新eb
	util.log_to_cmd("到达迭代终止条件{}",current_time)
	return m,bu,bi,P,Q

def matrix_factorization_base(data, f, times=500, t=1e-4, alpha=0.02, beta=0.02 ): #alpha=0.0002
	"""
	矩阵分解

	Args:
		data: 需要进行分解的矩阵R
		f: 隐含因子的个数
		e: 误差的预制，默认为0.001
		times: 最大迭代次数
		alpha: 学习率
		beta: 正则化系数
	Returns:
		返回P,Q矩阵

	"""
	util.log_to_cmd("开始运算")

	row_num, column_num=data.shape
	P=initNP(row_num, f) #用户潜在特征矩阵
	util.log_to_cmd("初始化P完毕")
	Q=initNQ(column_num, f) #项目潜在特征矩阵
	util.log_to_cmd("初始化Q完毕")
	data_nonzero=data.nonzero()
	data_zip=list(zip(data_nonzero[0],data_nonzero[1])) #zip 在py2中返回list,在py3中返回迭代器
	util.log_to_cmd("开始迭代")
	eb=0
	for current_time in range(times):
		util.log_to_cmd("第{}次迭代",current_time)
		e=0
		for item in range(len(data_zip)): #空读0.02 稀疏矩阵计算0.03603109688992845 数组计算0.0010
			x, y=data_zip[item]
			r=data[x,y]
			dot=np.dot(P[x,:],Q[:,y])
			#计算预测评分误差
			err=r-dot
			e+=pow(err,2)

			#更新P,Q
			for i in range(f):  
				P[x,i]+=alpha*(err*Q[i,y]-beta*P[x,i])
				Q[i,y]+=alpha*(err*P[x,i]-beta*Q[i,y])
				#计算损失函数
				e+=(beta/2)*(pow(P[x,i],2)+pow(Q[i,y],2))
		alpha*=0.9 #每次迭代缩小步伐
		util.log_to_cmd("损失值e={}",e)

       	#设置阈值停止迭代
		if np.abs(e-eb)<t:
			util.log_to_cmd("到达阈值终止条件{}",e)
			break
		eb=e #更新eb
	util.log_to_cmd("到达迭代终止条件{}",current_time)
	return P,Q


def testMat(mat): #太慢
	s=time.clock()
	c=0
	for i in range(len(mat)):
		for j in range(len(mat[0])):
			if mat[i][j]>0:
				mat[i][j]
				c+=1
				print(c)
	e=time.clock()
	util.log_to_cmd('一次迭代时间{}',e-s)  