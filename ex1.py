#!/usr/local/bin/python
import csv
import numpy as np
# import matplotlib.pyplot as plt

data_path = "mlclass-ex1-008/mlclass-ex1/ex1data1.txt"
f = open(data_path,"rb")

reader = csv.reader(f)
float_rows = [];
for row in reader:
	intermed = [];
	for i in row:
		intermed.append(float(i))
	float_rows.append(intermed)
	
	#Convert the data we just read in, in float_rows, to numpy arrays to plot.
data = np.array(float_rows)
data_size = data.shape
m = data_size[0]
n = data_size[1]
X = data[0:,0]
Y = data[0:,1]
#print(X.shape)

#plt.scatter(X,Y)
#plt.show()

############### Add Ones to data
ones_data = np.ones((data_size[0],data_size[1]+1))
ones_data[:,1:] = data

################ Normalize Features (do this step BEFORE you've added the ones)

# mu = np.mean(data,0);
# sigma = np.std(data,0);
# avg_X = np.dot(np.ones((m,1)),mu);
# std_X = np.dot(np.ones((m,1)),sigma);
# 
# X_norm = np.divide((data-avg_X),std_X);


############### Compute the Cost Function
y = np.reshape(data[0:,1],(data_size[0],1));

ones_data = ones_data[0:,0:2];
theta = np.zeros((data_size[1],1))


h_theta = np.dot(ones_data,theta)
diff = h_theta-y;
J = np.sum( np.multiply(diff,diff) )/(2*data_size[0])

#######Regularization

# #J += np.sum( np.multiply(theta[1:,1],theta[1:,1]) )*(LAMBDA/(2*m))
# #grad[1:,1] += (LAMBDA/m)*theta[1:,1]
# print(J)

################## Compute the gradient descent

alpha = 0.02
num_iters = 400
J_history = np.zeros((num_iters,1))
for iter in range(0,num_iters):
	h_theta = np.dot(ones_data,theta)
	diff = h_theta-y
	A=np.dot(diff,np.ones((1,n)))
	grad = np.resize(np.transpose(np.sum( np.multiply(ones_data,A),axis=0 )),(n,1) ) #to have $n+1 dimensional column vector
	grad /= m
	theta -= (alpha)*grad
	J_history[iter] = np.sum( np.multiply(diff,diff) )/(2*data_size[0])

print(theta)
### Plot data and linear fit
##Normal equations for (exact) solution to the linear linear fit here, uses the Moore-Penrose pseudo inverse
# theta = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(ones_data),ones_data)), np.transpose(ones_data) ),y)


# h_theta = np.dot(ones_data,theta);
# print(theta)
# #plt.plot(range(0,num_iters),J_history)
# plt.plot(X,Y,'yo',X,h_theta,'--k')
# plt.show()
