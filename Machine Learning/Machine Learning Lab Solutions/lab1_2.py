import numpy as np;


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

step=0.1;

theta1 = np.linspace(-10,10,num=(20/step)+1);
theta2=theta1;

x,y = np.meshgrid(theta1,theta2);

L = x**2 + y**2;

mini=1000000;
val=np.ones(2);

for i in range(len(theta1)):
	for j in range(len(theta2)):
		if(mini> L[i,j]):
			mini=L[i,j];
			val[0]=theta1[i];
			val[1]=theta2[j];

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, L);

plt.show();

