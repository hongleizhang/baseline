#encoding:utf-8

import numpy as np
import pandas as pd
import scipy.spatial.distance as ssd


#jaccard距离
def jaccard_distance(u, v):
	distance=ssd.jaccard(u,v)
	return distance

#余弦距离
def cosine_distance(u, v):
	distance=ssd.cosine(u, v)
	return distance

#欧式距离
def euclidean_distance(u, v):
	distance=ssd.euclidean(u, v)
	return distance

#皮尔逊相关系数
def pearson_distance():
	distance=ssd.correlation(u,v)
	return distance