import numpy as np;
import pandas as pd;
import math;
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 


df = pd.read_excel('data.xlsx');

n=len(df['x']);

x = np.array(df['x']).reshape((n,1));
y = np.array(df['y']);

mean=np.mean(x);
std = np.std(x);
x = (x-mean)/std;

step=0.1;

o = np.ones((len(x),1));
x = np.concatenate((o,x),axis=1);

theta0 = np.arange(-30,30,step);
theta1 = np.arange(-10,10,step);

t0,t1 = np.meshgrid(theta0,theta1);

theta = np.matmul(x.transpose(),x);
theta = np.linalg.inv(theta);

theta = np.matmul(theta,x.transpose());
theta = np.matmul(theta,y);

print(theta);

error=np.sum( (y - np.matmul(x,theta) )**2 );

theta[0]=23;
theta[1]=-7;
error1=np.sum( (y - np.matmul(x,theta) )**2 );

print(error,error1);







