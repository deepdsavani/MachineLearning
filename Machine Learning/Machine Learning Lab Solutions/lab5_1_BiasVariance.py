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
p = 10;
lim = lim*10; 

n = 1000;
x = np.linspace(0,lim,n);
x.resize((n,1));
y = np.zeros((n,1));


#-------------------linear regression----------

n_exp = 100;

fhats = np.zeros((n_exp,n,1));

for j in range(n_exp):

	for i in range(n):
		y[i] = func(x[i]);

	o = np.ones((n,1));
	xx = np.concatenate((o,x),axis=1);

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

bias = np.abs(f-mean_fhat);

variance = np.zeros((n,1));
for i in range(n):
	variance[i] = np.var( (fhats[:,i]-mean_fhat[i]));

mse = np.zeros((n,1));
for i in range(n):
	mse[i] = np.mean( (f[i]- fhats[:,i])**2);

# cnt = 0;
# for i in range(n):
# 	if(bias[i]**2 + variance[i] - mse[i] > (1e-3)):
# 		cnt+=1;
# print(cnt);

check = np.mean(bias**2 + variance);
check1 = np.mean(mse);

print("Mean of MSE",check1,"Bias^2 + variance",check);

print("Bias----",np.mean(bias) );
print("Variance-",np.mean(variance));

# print("Bias----",bias);
# print("MSE-----",mse);
# print("Variance-",variance);
# print("Bias^2 + variance",bias**2 + variance);

plt.plot(x,f)
plt.plot(x,mean_fhat)
plt.legend(['true','est'])
plt.show()





