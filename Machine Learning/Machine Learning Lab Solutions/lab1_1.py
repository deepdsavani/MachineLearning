import matplotlib.pyplot as plt;
import numpy as np;

step=0.1;

theta = np.linspace(-10,10,num=(20/step)+1);
L = theta*theta;

mini=1000;
val=1000;

for i in range(len(theta)):
	if(mini> L[i]):
		mini=L[i];
		ind=theta[i];

print(val);
plt.plot(theta,L);
plt.show();
