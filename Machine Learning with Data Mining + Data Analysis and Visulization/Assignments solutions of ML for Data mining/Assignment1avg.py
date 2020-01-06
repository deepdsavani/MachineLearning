import numpy as np
import matplotlib.pyplot as plt

t = 32
ANN = np.zeros((32,5));
LDA = np.zeros((32,5));
QDA = np.zeros((32,5));

with open("S1_ANN_6class.txt","r") as f1:
    
    cnt = 0
    for l in f1:
        
        ANN[cnt,0] = float(l)
        cnt+=1

with open("S2_ANN_6class.txt","r") as f2:
    
    cnt = 0
    for l in f2:
        
        ANN[cnt,1] = float(l)
        cnt+=1
    
with open("S3_ANN_6class.txt","r") as f3:
    
    cnt = 0
    for l in f3:
        
        ANN[cnt,2] = float(l)
        cnt+=1

with open("S4_ANN_6class.txt","r") as f4:
    
    cnt = 0
    for l in f4:
        
        ANN[cnt,3] = float(l)
        cnt+=1

with open("S5_ANN_6class.txt","r") as f5:
    
    cnt = 0
    for l in f5:
        
        ANN[cnt,4] = float(l)
        cnt+=1

#-----------------------------LDA-----------------------------------
with open("S1_LDA_6class.txt","r") as f1:
    
    cnt = 0
    for l in f1:
        
        LDA[cnt,0] = float(l)
        cnt+=1

with open("S2_LDA_6class.txt","r") as f2:
    
    cnt = 0
    for l in f2:
        
        LDA[cnt,1] = float(l)
        cnt+=1
    
with open("S3_LDA_6class.txt","r") as f3:
    
    cnt = 0
    for l in f3:
        
        LDA[cnt,2] = float(l)
        cnt+=1

with open("S4_LDA_6class.txt","r") as f4:
    
    cnt = 0
    for l in f4:
        
        LDA[cnt,3] = float(l)
        cnt+=1

with open("S5_LDA_6class.txt","r") as f5:
    
    cnt = 0
    for l in f5:
        
        LDA[cnt,4] = float(l)
        cnt+=1

#-----------------------------LDA-----------------------------------
with open("S1_QDA_6class.txt","r") as f1:
    
    cnt = 0
    for l in f1:
        
        QDA[cnt,0] = float(l)
        cnt+=1

with open("S2_QDA_6class.txt","r") as f2:
    
    cnt = 0
    for l in f2:
        
        QDA[cnt,1] = float(l)
        cnt+=1
    
with open("S3_QDA_6class.txt","r") as f3:
    
    cnt = 0
    for l in f3:
        
        QDA[cnt,2] = float(l)
        cnt+=1

with open("S4_QDA_6class.txt","r") as f4:
    
    cnt = 0
    for l in f4:
        
        QDA[cnt,3] = float(l)
        cnt+=1

with open("S5_QDA_6class.txt","r") as f5:
    
    cnt = 0
    for l in f5:
        
        QDA[cnt,4] = float(l)
        cnt+=1

avg_ANN = np.zeros(32)
avg_LDA = np.zeros(32)
avg_QDA = np.zeros(32)
time = np.zeros(32)

for i in range(ANN.shape[0]):
    
    time[i] = i+1
    avg_ANN[i] = np.mean(ANN[i,:])
    avg_LDA[i] = np.mean(LDA[i,:])
    avg_QDA[i] = np.mean(QDA[i,:])


plt.plot(time,avg_ANN)

plt.plot(time,avg_LDA)

plt.plot(time,avg_QDA)
plt.legend(["ANN","LDA","QDA"]);
plt.show()
