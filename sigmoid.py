#!/usr/local/bin/python

import numpy as np


def sigmoid(z):
	if isinstance(z,float) or isinstance(z,int):
		return 1/(1+np.exp(-z))
	else:
		return np.divide( np.ones(z.shape),(np.ones(z.shape)+np.exp(-z)) )