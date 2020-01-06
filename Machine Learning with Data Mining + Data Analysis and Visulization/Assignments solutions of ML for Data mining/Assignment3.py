import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import keras
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.utils import np_utils
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("creditcard.csv",delimiter=',')

y = np.array(df['Class'], dtype=np.int64)

print (y.shape)

df.drop(['Time','Class'], axis=1, inplace=True)

X = np.array(df)

print (X.shape)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print (x_train.shape, y_train.shape)
print (x_test.shape, y_test.shape)

#--------------------------------logistic regression----------------------------
clf = LogisticRegression(penalty='l2')

clf.fit(x_train,y_train)
acc = clf.score(x_test,y_test)

y_pred_LR = clf.predict(x_test)

print ("Logistic Regression: ", acc*100, "%");
# - 99.91397773954567 %


#--------------------------------Random Forest----------------------------------
clf2 = RandomForestClassifier(n_estimators=21,criterion='entropy',min_samples_split=0.05, max_depth=10);

clf2.fit(x_train, y_train)

acc2 = clf2.score(x_test,y_test)

y_pred_RF = clf2.predict(x_test)

print ("Random Forest: ", acc2*100, "%");
# -  99.91748885221726 %
#--------------------------------ANN--------------------------------------------

model=Sequential()
model.add(Dense(256,input_shape=(x_train.shape[1],),activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['binary_accuracy']);
history=model.fit(x_train,y_train,batch_size=64,epochs=10,verbose=0)
scores=model.evaluate(x_test,y_test)
y_pred_NN = model.predict(x_test)

y_pred = np.zeros(y_pred_LR.shape[0], dtype=np.int64)

print ("ANN: ", scores[1]*100, "%");
# 99.9385555282469 %


cnt = 0
for i in range(y_pred_LR.shape[0]):
    
    if (y_pred_LR[i] + y_pred_RF[i] + y_pred_NN[i]) >= 2:        
        y_pred[i] = 1   
    else:       
        y_pred[i] = 0 
    
    if y_pred[i] == y_test[i]:        
        cnt+=1

print ("Ensemble: ", (float(cnt)/float(y_pred_LR.shape[0]))*100, "%");
# - 99.90871107053826 %