#!/usr/local/bin/python

import numpy as np

for i in range(0,1000):
	X = np.array(X);
	y = np.array(y);
        tup1 = X.shape;
        m = tup1[0];
        n = tup1[1];
	theta = np.array(theta_list);
	h_theta = X.dot(theta);
	diff = h_theta-y;
	J = diff.dot(diff)/m;
	A = diff*(np.ones((1,n)));
	grad = np.transpose(X.dot(A));

	theta = theta - (alpha)*grad;

theta_out = theta;

% function [J,grad] = lin_regress_cost(X,y,theta)
% 	m = size(X,1);
% 	A=diff*ones(1,n); %makes m x n dimensional matrix, all have same columns.
% 
% 	grad = sum(X.*A)';  % Sum returns a row vector, take transpose to have $n+1 dimensional column vector
% 	grad = grad/m;
% 
% end