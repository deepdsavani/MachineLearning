import numpy as np;
import matplotlib.pyplot as plt;
import math;

n=1000;
m=2;
lim1=5;
lim=1000;

x = np.zeros((n,m) );
y = np.zeros((n,1));

var=100;

for i in range(int(n/2) ):
	for j in range(m):
		# mean = (lim/(2*m) )*j;
		# x[i,j]=np.random.normal(mean,var);
		# x[i,j] = (x[i,j]-mean)/math.sqrt(var);
		x[i,j] = np.random.normal(-1,0.5);
	y[i] = 0;

for i in range(int(n/2) ,n,1):
	for j in range(m):
		
		# x[i,j]=np.random.normal(,var );
		x[i,j] = np.random.normal(1,0.5);
	y[i]=1;

lr=1e-2;


pred=np.random.uniform(0,lim1,m);

dtheta=1e-5;

epoch=1000;

for i in range(epoch):
	y_pred = np.matmul(x,pred);
	# print(y_pred);
	y_pred = 1/(1 + np.exp(-y_pred));

	# print(y_pred);


	for j in range(m):
		p_diff=0;
		for k in range(n):
			p_diff+= x[k,j]*(y[k]-y_pred[k]);
		p_diff*=(-2.0/n);
		pred[j] = pred[j] - lr*p_diff;


y_pred = np.matmul(x,pred);
y_pred = 1/(1 + np.exp(-y_pred));

error=0;
for i in range(n):
	if(y_pred[i]<=0.5):
		y_pred[i]=0;
	else:
		y_pred[i]=1;

	if(y_pred[i]!=y[i]):
		error+=1;

# plt.scatter(x[:,0],x[:,1]);
# plt.show();

c = ['b','g'];
for i in range(n):
	plt.scatter(x[i,0],x[i,1],color=c[int(y_pred[i])]);
plt.show();
print(error/len(y_pred));
print(pred);



