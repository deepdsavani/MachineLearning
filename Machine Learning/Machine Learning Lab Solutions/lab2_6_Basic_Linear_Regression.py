import numpy as np;
import pandas as pd;
import math;
import random;
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 


df = pd.read_excel('data.xlsx');

x = np.array(df['x']);
y = np.array(df['y']);

mean=np.mean(x);
std = np.std(x);
x = (x-mean)/std;

theta0 = np.random.normal();
theta1 = np.random.normal();

prev0 = theta0 + 1;
prev1 = theta1 + 1;

eps = 1e-6;
alpha = 0.01;

epoch = 0;

while(abs(prev0-theta0) >=eps or abs(prev1-theta1) >=eps):

	prev0 = theta0;
	prev1 = theta1;


	ind = np.random.randint(0,len(x)-1);

	error0 = -2*(y[ind] - (theta0 + theta1*x[ind]));
	error1 = -2*(y[ind] - (theta0 + theta1*x[ind]))*x[ind];

	theta0 = theta0 - alpha*(error0);
	theta1 = theta1 - alpha*(error1);

print(theta0,theta1);