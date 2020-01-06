import numpy as np;
import matplotlib.pyplot as plt;

theta = np.random.normal();

prev = theta + 1;
eps = 1e-6;
alpha = 0.1;

epoch = 0;

while( abs(theta-prev) >= eps):

	prev = theta;

	alpha = alpha - (-4*(prev-1)*(prev - 2*alpha*(prev-1) - 1 ) )/( (8* ((prev-1)**2) ) + eps );
	print(alpha);

	theta = theta - alpha*(2*(theta-1));
	epoch+=1;

print(epoch);
print(theta);