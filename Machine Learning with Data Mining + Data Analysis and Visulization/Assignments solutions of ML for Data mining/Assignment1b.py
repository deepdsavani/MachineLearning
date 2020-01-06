import scipy.io
import numpy as np
import pandas as pd
# import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.utils import np_utils
from keras import backend as K
from imutils import build_montages
import matplotlib.pyplot as plt
import numpy as np
# import cv2
from keras.models import Sequential
from keras.layers import Dense
import keras

mat = scipy.io.loadmat('SUB_100307_S2.mat')

#print mat['X_3D']  --> numpy array size - (124, 32, 5188)

X = mat['D'] # (240, 2035, 124)

X = X.reshape(X.shape[0],X.shape[1]*X.shape[2]); # (240, 252340)

for i in range(X.shape[0]):
    u = np.mean(X[i,:]);
    std = np.std(X[i,:]);
    X[i,:] = (X[i,:]-u)/std;

X = X.transpose();

#---------------------------Code for PCA-----------------

pca = PCA(0.95)

pca.fit(X);

X_regen = pca.transform(X);

print(X_regen.shape);

m = X_regen.shape[1]; # 74

X_transform = pca.inverse_transform(X_regen);

mse = np.mean( (X-X_transform)**2 );

print(mse); #--------------0.04943373762041574

#------------------------Code for auto encoder

features = X.shape[1];

model=Sequential()
model.add(Dense(m,input_shape=(features,),activation='relu'))
model.add(Dense(features,activation='linear'))
model.summary()
model.compile(loss='mse',optimizer='adam');
history=model.fit(X,X,batch_size=1024,epochs=20,verbose=1)

#---------------------loss mse - 0.0644--------------------------------