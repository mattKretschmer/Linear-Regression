#!/usr/local/bin/python
import csv
import numpy as np


def import_data( path):
	f = open(path,"rb")
	reader = csv.reader(f)
	float_rows = []
	for row in reader:
		intermed = []
		for i in row:
			intermed.append(float(i))
		float_rows.append(intermed)	
	data = np.array(float_rows)
	return data

def process_data(raw_data):
# Adds a column of ones to the data, and then splits into X and y
	data_size = raw_data.shape
	m = data_size[0]
	n = data_size[1]
	ones_data = np.ones((m,n+1))
	ones_data[:,1:] = raw_data
	print ones_data
	y = np.reshape(raw_data[0:,-1],(m,1))
	ones_X = ones_data[0:,0:-1]
	return ones_X,y

	
def feature_normalize(data):
# 	Important, remember to do this BEFORE adding the column of ones!
	mu = np.mean(data,0)
	sigma = np.std(data,0)
	avg_X = np.dot(np.ones((m,1)),mu)
	std_X = np.dot(np.ones((m,1)),sigma)
	
	X_norm = np.divide((data-avg_X),std_X)
	return mu,sigma,X_norm
	
def costJ(X,y,theta):
	data_size = X.shape
	m = data_size[0]
	n = data_size[1]
	h_theta = np.dot(X,theta)
	diff = h_theta-y
	J = np.sum( np.multiply(diff,diff) )/(2*data_size[0])
	A=np.dot(diff,np.ones((1,n)))
	grad = np.resize(np.transpose(np.sum( np.multiply(X,A),axis=0 )),(n,1) ) #to have $n+1 dimensional column vector
	grad /= m
	return J,grad
	
def costJ_reg(X,y,theta,LAMBDA):
	data_size = X.shape
	m = data_size[0]
	n = data_size[1]
	h_theta = np.dot(X,theta)
	diff = h_theta-y
	J = np.sum( np.multiply(diff,diff) )/(2*data_size[0])+np.sum( np.multiply(theta[1:,1],theta[1:,1]) )*(LAMBDA/(2*m))
	A=np.dot(diff,np.ones((1,n)))
	grad = np.resize(np.transpose(np.sum( np.multiply(X,A),axis=0 )),(n,1) ) 
	grad /= m
	grad[1:,1] += (LAMBDA/m)*theta[1:,1]
	return J,grad
	
def grad_descent(X,y,theta,alpha,num_iters):
	J_history = np.zeros((num_iters,1))
	for iter in range(0,num_iters):
		J,grad = costJ(X,y,theta)
		theta -= (alpha)*grad
		J_history[iter] = J
	return theta,J_history
	
def grad_descent_reg(X,y,theta,LAMBDA,alpha,num_iters):
	J_history = np.zeros((num_iters,1))
	for iter in range(0,num_iters):
		J,grad = costJ_reg(X,y,theta,LAMBDA)
		theta -= (alpha)*grad
		J_history[iter] = J
	return theta,J_history
	
def normal_Equations(X,y):
	theta = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(X),X)), np.transpose(X) ),y)
	pinv(X'*X)*X'*y
	return theta