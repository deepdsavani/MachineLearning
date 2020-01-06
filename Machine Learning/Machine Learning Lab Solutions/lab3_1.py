import numpy as np;
import pandas as pd;
import math;
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 


df = pd.read_excel('ex2data1-logistic.xls');


x1 = np.array(df['x1']);
x2 = np.array(df['x2']);
y = np.array(df['y']);

c = ['b','g'];

for i in range(len(x1)):
	plt.scatter(x1[i],x2[i],color = c[y[i]]);

plt.show();

