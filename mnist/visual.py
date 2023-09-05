import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_train=pd.read_csv(r'E:\dev\sequential\mnist\mnist_train.csv')
data_test=pd.read_csv(r'E:\dev\sequential\mnist\mnist_test.csv')

inputs_train=data_train.iloc[:, 1:].values
targets_t=data_train.iloc[:, 0].values
targets_train= np.full((targets_t.shape[0], 10), 0.01)  
for i in range(targets_t.shape[0]):
    targets_train[i][targets_t[i]]=0.99
    
inputs_test=data_test.iloc[:, 1:].values
targets_s=data_test.iloc[:, 0].values
targets_test= np.full((targets_s.shape[0], 10), 0.01)  
for i in range(targets_s.shape[0]):
    targets_test[i][targets_s[i]]=0.99

for i in range(10):    
    print(targets_s[i])
    image_array1=np.asfarray(inputs_test[i]).reshape((28,28))
    plt.figure(figsize=(3,3))
    plt.imshow(image_array1,cmap='Greys',interpolation='None')
    plt.show()
   
for i in range(10):    
    print(targets_t[i])
    image_array2=np.asfarray(inputs_train[i]).reshape((28,28))
    plt.figure(figsize=(3,3))
    plt.imshow(image_array2,cmap='Greys',interpolation='None')
    plt.show()