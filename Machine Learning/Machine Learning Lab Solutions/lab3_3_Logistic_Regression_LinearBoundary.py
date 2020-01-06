import numpy as np;
import pandas as pd;
import math;
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

from sklearn.model_selection import train_test_split


df = pd.read_excel('ex2data1-logistic.xls');

y = df['y'];
df.drop(['y'], axis=1, inplace = True);

x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.1, random_state=42)


theta = np.random.normal(0,1,3);

x1 = x_train['x1'].values;
x2 = x_train['x2'].values;

y = y_train.values;

n = len(x_train);
one = np.ones((n,1));
x1 = x1.reshape((n,1));
x2 = x2.reshape((n,1));

x = np.concatenate((x1,x2),axis=1);
x = np.concatenate((one,x),axis=1);

prev = np.zeros(3);
prev = theta + 1;

epoch = 0;
alpha = 0.001;
eps = 1e-6;

while( epoch< 5e5):

	y_pred = np.matmul(x,theta);
	y_pred = 1.0/(1.0 + np.exp(-y_pred));

	err1 = (y_pred - y_train);

	for i in range(3):
		err = np.sum(err1*(x[:,i]));
		theta[i] = theta[i] - alpha*err;

	# print(np.sum((y_pred- y_train)));
	epoch+=1;

test = 0;

x1_test = x_test['x1'].values;
x2_test = x_test['x2'].values;

y_test = y_test.values;

n = len(x_test);
one = np.ones((n,1));
x11 = x1_test.reshape((n,1));
x21 = x2_test.reshape((n,1));

x = np.concatenate((x11,x21),axis=1);
x = np.concatenate((one,x),axis=1);

y_pred = np.matmul(x,theta);
y_pred = 1.0/(1.0 + np.exp(-y_pred));

for i in range(len(y_pred)):
	if(y_pred[i]>=0.5):
		y_pred[i]=1;
	else:
		y_pred[i]=0;
	if(y_pred[i]==y_test[i]):
		test+=1;


print("test accuracy - ",(test*100)/len(y_pred));

min_X1 = min(np.min(x1),np.min(x11))
max_X1 = max(np.max(x1),np.max(x11))

A = np.linspace(min_X1,max_X1,num=100)
B = (-theta[0] - (theta[1]*A))/theta[2];

c = ['b', 'g']

for i in range(len(x1)):
    
    plt.scatter(x1[i], x2[i], color=c[int(y[i])])

for i in range(len(x11)):
    
    plt.scatter(x11[i], x21[i], color=c[int(y_test[i])])

plt.plot(A,B)
plt.show()

print(theta);
print(epoch);
