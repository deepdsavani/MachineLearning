import numpy as np;
import pandas as pd;
import math;
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

def func(x):
	ans = x + 2*math.sin(1.5*x) + np.random.normal(0,np.sqrt(2) );
	return ans;
def true_func(x):
	ans = x + 2*math.sin(1.5*x);
	return ans;

lim = (2*np.pi)/1.5;
p = 3;
lim = lim*p; 

n = 1000;
x = np.linspace(0,lim,n);
x.resize((n,1));
y = np.zeros((n,1));


#-------------------linear regression----------

n_exp = 1000;

fhats = np.zeros((n,n,1));

for j in range(n_exp):

	for i in range(n):
		y[i] = func(x[i]);

	o = np.ones((n,1));
	xx = np.concatenate((o,x),axis=1);

	for k in range(2,10,1):
		xx = np.concatenate((xx,x**k),axis=1);
		


	theta = np.matmul(xx.transpose(),xx);
	theta = np.linalg.inv(theta);

	theta = np.matmul(theta,xx.transpose());
	theta = np.matmul(theta,y);

	y_pred = np.matmul(xx,theta);

	# print(fhats[j].shape);
	fhats[j] = y_pred;
	# print(theta);

mean_fhat = np.zeros((n,1));
for i in range(n):
	mean_fhat[i] = np.mean(fhats[:,i]);

plt.plot(x,y);
plt.plot(x,mean_fhat);
plt.show();

f = np.zeros((n,1));
for i in range(n):
	f[i] = true_func(x[i]);

bias =np.abs(f-mean_fhat);

variance = np.zeros((n,1));
for i in range(n):
	variance[i] = np.mean((fhats[i,:]-mean_fhat)**2);

mse = np.zeros((n,1));
for i in range(n):
	mse[i] = np.mean( (f- fhats[i,:])**2);

check = np.mean(bias**2 + variance);
check1 = np.mean(mse);

print("Mean of MSE",check1,"Bias^2 + variance",check);

# print("Bias----",bias);
# print("MSE-----",mse);
# print("Variance-",variance);
# print("Bias^2 + variance",bias**2 + variance);





