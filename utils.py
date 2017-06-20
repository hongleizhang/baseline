#encoding:utf-8

import time
import numpy as np
import pandas as pd


# 向控制台输出日志
def log_to_cmd(msg,spend_time=None,time_from=None,time_to=None):
	"""
		向控制台输出日志
	"""
	if spend_time!=None:
		print msg,spend_time
		return
	if time_from!=None:
		print msg,time_from,time_to
		return
	print msg

def function():
	pass
