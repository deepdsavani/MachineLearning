import h5py

import numpy as np
import matplotlib.pyplot as plt

mat = h5py.File('faceimages.mat')

data = list(mat['data'])
data = np.array(mat['data'], dtype=np.float64)

print(data.shape);

X = data[:-1,:]

c = np.cov(X)

print (c.shape)

w, v = np.linalg.eigh(c)

print (v.shape)

Y = np.matmul(v.transpose(),X)

print (Y.shape)

Y_50 = Y[-50:,:]
Y_100 = Y[-100:,:]
Y_400 = Y[-400:,:]

X_50 = np.matmul(v[:,-50:],Y_50)
X_100 = np.matmul(v[:,-100:],Y_100)
X_400 = np.matmul(v[:,-400:],Y_400)

mse50 = np.mean((X-X_50)**2)
mse100 = np.mean((X-X_100)**2)
mse400 = np.mean((X-X_400)**2)

print (mse50)
print (mse100)
print (mse400)

orig_image = X[:,0].reshape(92,112)
recon_image_50 = X_50[:,0].reshape(92,112)
recon_image_100 = X_100[:,0].reshape(92,112)
recon_image_400 = X_400[:,0].reshape(92,112)

# plt.imshow(orig_image)
# plt.figure()
# plt.imshow(recon_image_50)
# plt.figure()
# plt.imshow(recon_image_100)
# plt.figure()
# plt.imshow(recon_image_400)
# plt.show()