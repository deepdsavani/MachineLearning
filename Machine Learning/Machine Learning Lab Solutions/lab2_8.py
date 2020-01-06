import numpy as np;
import matplotlib.pyplot as plt;

theta1 = np.random.normal();
theta2 = np.random.normal();

prev1 = theta1 + 1;
prev2 = theta2 + 1;

eps = 1e-6;
alpha = 0.1;

epoch = 0;

while( abs(theta1-prev1) >= eps or abs(theta2 - prev2)>=eps ):

	prev1 = theta1;
	prev2 = theta2;

	alpha = alpha - ( (-4*prev1*(prev1 - 2*alpha*prev1)) + (-4*prev2*(prev2 - 2*alpha*prev2)) )  / ( (8*prev1*prev1) + (8*prev2*prev2) + eps );
	print(alpha);

	theta1 = theta1 - alpha*(2*theta1);
	theta2 = theta2 - alpha*(2*theta2);
	epoch+=1;

print(epoch);
print(theta1,theta2);