import numpy as np;
import pandas as pd;
import math;
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import tensorflow as tf

from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data();

print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)
print("x_train shape:", x_test.shape, "y_train shape:", y_test.shape)


IMAGE_PIXELS = x_train.shape[1]*x_train.shape[2];

x_train = x_train.reshape(x_train.shape[0],IMAGE_PIXELS);
x_test = x_test.reshape(x_test.shape[0],IMAGE_PIXELS);

y_train = y_train.reshape( (y_train.shape[0],1));
y_test = y_test.reshape( (y_test.shape[0],1));


n_classes = 2;

temp = [];
temp1 =[];
for i in range(len(x_train)):
	if(y_train[i]<n_classes):
		temp.append(x_train[i]);
		temp1.append(y_train[i]);
x_train = temp;
y_train = temp1;

temp = [];
temp1 = [];
for i in range(len(x_test)):
	if(y_test[i]<n_classes):
		temp.append(x_test[i]);
		temp1.append(y_test[i]);
x_test = temp;
y_test = temp1;

print(len(x_test),len(x_train));
print(len(y_test),len(y_train));

#-----calculcation of pi

pi = np.zeros((n_classes,1));
tot=0;

for i in range(len(x_train)):
	pi[y_train[i]]+=1;
	tot+=1;
pi/=tot;

print(pi);

#-------------------Calculation of Mu[i,j] i-class,j-feature --------------------

Dict = {};
for i in range(n_classes):
	Dict[i] = [];

for i in range(len(x_train)):
	Dict[int(y_train[i]) ].append(x_train[i]);

for i in range(n_classes):
	Dict[i] = np.array(Dict[i]);

mu = np.zeros((n_classes,IMAGE_PIXELS));

for i in range(n_classes):
	for j in range(IMAGE_PIXELS):
		mu[i,j] = np.mean(Dict[i][:,j]);

#--------------Covariance matrix calculated--------

cov = np.zeros((n_classes,IMAGE_PIXELS,IMAGE_PIXELS));

for i in range(n_classes):
	cov[i] = np.cov(Dict[i].T);

cov1 = np.zeros((IMAGE_PIXELS,IMAGE_PIXELS));
for i in range(IMAGE_PIXELS):
	for j in range(IMAGE_PIXELS):
		temp = 0;
		for k in range(n_classes):
			temp+=cov[k,i,j];
		temp/=n_classes;
		cov1[i,j] = temp;

#------------------------------------------------


eps = 0.00001;
den = 1.0/ ( np.sqrt(np.linalg.det(cov1)) + eps ) ;
den = np.log(den);


covinv = np.linalg.pinv(cov1);


acu=0;
for i in range(len(x_test)):	
	prob = 0;
	max_class = 0;
	for j in range(n_classes):

		sample = x_test[i];
		sample = sample - mu[j,:];
		temp = np.matmul( np.matmul(sample.T,covinv) ,sample);


		temp = -0.5*temp;
		temp += np.log(pi[j]);
		temp += den;
		# print(den.shape);
		# print(temp.shape);
		if(j==0):
			prob = temp;
		elif(temp > prob):
			max_class = j;
			prob = temp;
	if(y_test[i]==max_class):
		acu+=1;

print( (acu*100)/len(y_test));