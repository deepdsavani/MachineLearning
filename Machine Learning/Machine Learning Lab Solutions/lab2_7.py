+import numpy as np;
import matplotlib.pyplot as plt;

theta = np.random.normal();

prev = theta + 1;
eps = 1e-6;
alpha = 0.1;

epoch = 0;

while( abs(theta-prev) >= eps):

	prev = theta;

	alpha = alpha - (-4*prev*(prev - 2*alpha*prev))/((8*prev*prev) + eps );
	print(alpha);

	theta = theta - alpha*(2*theta);
	epoch+=1;

print(epoch);
print(theta);