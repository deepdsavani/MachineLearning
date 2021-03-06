import scipy.io
import numpy as np
import pandas as pd
import matplotlib as plt
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# import cv2
from keras.models import Sequential
from keras.layers import Dense
import keras


mat = scipy.io.loadmat('S5.mat')

#print mat['X_3D']  --> numpy array size - (124, 32, 5188)

X = mat['X_3D']

Y = mat['categoryLabels']

# print (X.shape, Y.shape)

Y = Y.transpose()

X = X.transpose(1, 2, 0)

# print (X.shape, Y.shape)  # --> X - (32 - time , 5188, 124 - channels)

#print y

#----------------------z normalization-------------------
for i in range(X.shape[0]):
	for j in range(X.shape[2]):
		mu = np.mean(X[i,:,j]);
		std = np.std(X[i,:,j]);
		X[i,:,j] = (X[i,:,j]-mu)/std;

LDA = np.zeros(X.shape[0]);
QDA = np.zeros(X.shape[0]);
ANN = np.zeros(X.shape[0]);
Time = np.zeros(X.shape[0]);

pca = PCA(0.95)

for j in range(X.shape[0]):
    
    pca.fit(X[j,:,:])

    X_mod = pca.transform(X[j,:,:])
    
    # print (X_mod.shape)
    
    data_X = []
    data_Y = []
    
    for i in range(X.shape[1]):
        
        #print Y[i,0]
        if Y[i]==2 or Y[i]==6 :
            
            data_X.append(X[j,i,:])
            if Y[i]==2:
                data_Y.append(0)
            else:
                data_Y.append(1)
    
    sz = len(data_Y)
    data_X = np.array(data_X, dtype=np.float64)
    data_Y = np.array(data_Y, dtype=np.int64).reshape((sz,1))
    
    print (data_Y.shape)
    
    x_train, x_test, Y_train, Y_test = train_test_split(data_X, data_Y, test_size=0.2)

    y_train = []
    y_test = []

    y_train = np.array(Y_train, dtype=np.float64)
    y_test = np.array(Y_test, dtype=np.float64)

    # 2-class classification
    
    for i in range(x_train.shape[1]):
        
        x_train[:,i] = (x_train[:,i] - np.mean(x_train[:,i]))/ np.std(x_train[:,i])

    #------------------------_Code for LDA--------------------
    clf = LinearDiscriminantAnalysis();
    clf.fit(x_train,y_train);
    
    LDA_acc = clf.score(x_test, y_test);    
    print ("LDA accuracy: ", LDA_acc);
    
    LDA[j] = LDA_acc*100;

    #------------------------ Code for QDA------------------------------
    clf2 = QuadraticDiscriminantAnalysis()
    clf2.fit(x_train,y_train)
    
    QDA_acc = clf2.score(x_test, y_test)
    print ("QDA accuracy: ", QDA_acc)
    
    QDA[j] = QDA_acc*100;

    #----------------------------------------

    features = x_train.shape[1];

    model=Sequential()
    model.add(Dense(64,input_shape=(features,),activation='relu'))
    model.add(Dense(32,input_shape=(features,),activation='relu'))
    model.add(Dense(16,input_shape=(features,),activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['binary_accuracy'])
    history=model.fit(x_train,y_train,batch_size=32,epochs=20,verbose=0)
    scores=model.evaluate(x_test,y_test)


    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    Time[j] = j+1;
    ANN[j] = scores[1]*100;   

with open("S5_ANN.txt","w") as f1:
    
    for x in ANN:
        f1.write(str(x))
        f1.write("\n")    
    f1.close()

with open("S5_LDA.txt","w") as f2:
    
    for x in LDA:
        f2.write(str(x))
        f2.write("\n")
    f2.close()

with open("S5_QDA.txt","w") as f3:
    
    for x in QDA:
        f3.write(str(x))
        f3.write("\n")    
    f3.close()

plt.plot(Time,ANN)
plt.title("ANN accuracy")
plt.figure()
plt.plot(Time,LDA)
plt.title("LDA accuracy")
plt.figure()
plt.plot(Time,QDA)
plt.title("QDA accuracy")
plt.show()

