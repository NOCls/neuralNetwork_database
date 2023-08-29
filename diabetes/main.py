from sklearn .model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#personal_difine_file
import utils 
import visual

df = pd.read_csv('E:\dev\MLP_c02\diabetes.csv')
df = utils.preprocess(df)
#visual.visual()

X = df.loc[:,df.columns != 'Outcome']
y = df.loc[:,'Outcome']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size = 0.2)

model = Sequential()
model.add(Dense(32,activation='relu',input_dim = 8))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=200)

scores = model.evaluate(X_train,y_train)
print("training accuracy:%.2f%%\n"%(scores[1]*100))

scores = model.evaluate(X_test,y_test)
print("testing accuracy:%.2f%%\n"%(scores[1]*100))

y_test_pred = model.predict(X_test)
y_test_pred_classes = (y_test_pred > 0.5).astype(int)

c_matrix = confusion_matrix(y_test, y_test_pred_classes)
ax = sns.heatmap(c_matrix,annot=True, xticklabels=['No Diabetes', 'Diabetes'], 
                 yticklabels=['No Diabetes', 'Diabetes'], cbar=False, cmap='Blues')
ax.set_xlabel("Prediction")
ax.set_ylabel("Actual")
plt.show()

y_test_pred_probs = model.predict(X_test)
FPR,TPR,_ = roc_curve(y_test,y_test_pred_probs)

plt.plot(FPR,TPR)
plt.plot([0,1],[0,1],'--',color ='black')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

