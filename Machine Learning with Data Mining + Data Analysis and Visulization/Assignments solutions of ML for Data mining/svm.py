import scipy.io
import numpy as np
import pandas as pd
#import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt;
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

mat = scipy.io.loadmat('S1.mat')

#print mat['X_3D']  --> numpy array size - (124, 32, 5188)

X = mat['X_3D']

Y = mat['categoryLabels']

#print X.shape, y.shape

Y = Y.transpose()

X = X.transpose(1, 2, 0)

print(X.shape, Y.shape)  # --> X - (32 - time , 5188, 124 - channels)

#print y

pca = PCA(0.95)

final=[]
y=[]
for j in range(X.shape[0]):
    
    pca.fit(X[j,:,:])

    X_mod = pca.transform(X[j,:,:])
    
    data_X = []
    data_Y = []
    
    for i in range(X.shape[1]):
        
        #print Y[i,0]
        if Y[i,0]==2 or Y[i,0]==6 :
            
            data_X.append(X[j,i,:])
            if Y[i,0]==2:
                data_Y.append(0)
            else:
                data_Y.append(1)
    
    data_X = np.array(data_X, dtype=np.float64)
    data_Y = np.array(data_Y, dtype=np.int64).reshape((len(data_Y),1))
    
    x_train, x_test, Y_train, Y_test = train_test_split(data_X, data_Y, test_size=0.1)

    svclassifier = SVC(kernel='rbf')
    svclassifier.fit(x_train, Y_train.ravel())
    y_pred = svclassifier.predict(x_test)
    
    cnt=0;
    for i in range(y_pred.shape[0]):
        if(y_pred[i]==Y_test[i]):
            cnt=cnt+1
    print(cnt/y_pred.shape[0]);

    final.append(cnt/y_pred.shape[0]);
    y.append(j+1)

plt.plot(y,final)
plt.show()
    
