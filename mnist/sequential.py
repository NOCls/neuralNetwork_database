from keras.models import Sequential
from keras.layers import Dense
from keras import initializers, optimizers 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

model=Sequential()
initializer =initializers.RandomNormal(mean=0.0, stddev=0.04)  
model.add(Dense(units=100,activation='sigmoid',input_dim=784,kernel_initializer=initializer))
model.add(Dense(units=100,activation='sigmoid'))
model.add(Dense(units=10,activation='sigmoid'))

sgd_optimizer = optimizers.SGD(learning_rate=0.25)
model.compile(loss='mean_squared_error', optimizer=sgd_optimizer, metrics=['accuracy'])
model.fit(inputs_train,targets_train,epochs=7,verbose=2)

print("-------------------------------")
scores_train =model.evaluate(inputs_train,targets_train,verbose=2)
print("traing accuracy:",scores_train[1])
scores_test =model.evaluate(inputs_test,targets_test,verbose=2)
print("test accuracy:",scores_test[1])

