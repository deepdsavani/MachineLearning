 import numpy as np;
import pandas as pd;
import math;
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 


df = pd.read_excel('data.xlsx');

x = np.array(df['x']);
y = np.array(df['y']);

mean=np.mean(x);
std = np.std(x);
x = (x-mean)/std;

step=0.1;

theta0 = np.arange(-30,30,step);
theta1 = np.arange(-10,10,step);


t0,t1 = np.meshgrid(theta0,theta1);
L = np.zeros(t0.shape);


mini = np.inf;
val= np.zeros(2);

for i in range(len(theta1)):
	for j in range(len(theta0)):
		error=0;
		for k in range(len(x)):
			error+=(y[k]- (t0[i,j] + t1[i,j]*x[k]) )**2;
		L[i,j]=error;

		if(error<mini):
			mini=error;
			val[0]=t0[i,j];
			val[1]=t1[i,j];

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(t0, t1, L);


print(val);

plt.show();







