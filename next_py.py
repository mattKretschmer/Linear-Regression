#!/usr/local/bin/python
import linear_regression
import numpy as np

path = "mlclass-ex1-008/mlclass-ex1/ex1data1.txt" #Local path to ML-Learning data

raw_data = linear_regression.import_data( path)
print raw_data
X,y = linear_regression.process_data(raw_data)
data_size = X.shape
n = data_size[1]
print X
###gradient descent

num_iters = 400
alpha = 0.02

theta = np.zeros((n,1))
theta,J_history=linear_regression.grad_descent(X,y,theta,alpha,num_iters)
# theta,J_history =linear_regression.grad_descent_reg(X,y,theta,LAMBDA,alpha,num_iters)
print theta
### Learn from Normal Equations too

theta = linear_regression.normal_Equations(X,y)
print theta