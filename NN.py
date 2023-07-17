
import tensorflow as tf
import math

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
import math
import pickle
import pandas as pd
import seaborn

import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from matplotlib import pyplot as plt

from sklearn.metrics import mean_squared_error,mean_absolute_error

df1 = pd.read_csv('features_short.csv')
df2 = pd.read_csv('labels_short.csv')

X_train,X_test,y_train,y_test = train_test_split(df1,df2,test_size=0.2)

X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)


model = Sequential()
model.add(Dense(700,input_shape=(5,),activation='relu'))
model.add(Dense(700,activation='relu'))
model.add(Dense(700,activation='relu'))
model.add(Dense(700,activation='relu'))
#model.add(Dense(400,activation='relu'))
model.add(Dense(1,))
model.compile(Adam(lr=0.0002),'mean_squared_error')
earlystopper = EarlyStopping(monitor='val_loss',min_delta=0,patience=100,verbose=1,mode='auto')

history = model.fit(X_train,y_train,epochs=1000,validation_split=0.2,shuffle=True,verbose=2, callbacks= [earlystopper])

history_dict=history.history

loss_value=history_dict['loss']
val_loss_values=history_dict['val_loss']
plt.figure()
plt.plot(loss_value,'bo',label='training_loss')
plt.plot(val_loss_values,'r',label='val training loss')

y_train_pred= model.predict(X_train)
y_test_pred=model.predict(X_test)

model.save("NN_1.h5")

print ("y_test",y_test )
print ("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
print("y_test_pred",y_test_pred)
print ("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

print('MSE: %.2f'% mean_squared_error(y_test, y_test_pred))
print('RMSE  of testing : %.2f' %(math.sqrt(mean_squared_error(y_test, y_test_pred))))
print('MAE  testing set: %.2f'% mean_absolute_error(y_test, y_test_pred))
print("the R2 score on the test set is :\t{:0.3f}".format(r2_score(y_test,y_test_pred)))