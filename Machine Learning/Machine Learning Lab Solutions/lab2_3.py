import numpy as np;
import matplotlib.pyplot as plt;

theta = np.random.normal();

prev = theta + 1;
eps = 1e-6;
alpha = 0.1;

epoch = 0;

while( abs(theta-prev) >= eps):

	prev = theta;
	theta = theta - alpha*(2*(theta-1) );
	epoch+=1;

print(epoch);
print(theta);