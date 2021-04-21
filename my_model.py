import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import r2_score
from numpy.random import seed
seed(1)
seed(2)


dataset = pd.read_csv('admissions_data.csv')

feature = dataset.iloc[:,1:-1]
labels = dataset.iloc[:,-1]

feature_train, feature_test, labels_train, labels_test = train_test_split(feature, labels, test_size=0.33, random_state=23)


feature_train_scaled = StandardScaler().fit_transform(feature_train)
feature_test_scaled = StandardScaler().fit_transform(feature_test)

my_model = Sequential()
input = InputLayer(input_shape = feature.shape[1],)
my_model.add(input)
my_model.add(Dense(64, activation='relu'))
my_model.add(Dense(16,activation='relu'))

my_model.add(Dense(1))
print(my_model.summary())

opt = Adam(learning_rate=0.1)

my_model.compile(loss='mse',metrics=['mae'], optimizer = opt)

stop = EarlyStopping(monitor = 'val_mae', mode='min',verbose=1, patience=5)

history = my_model.fit(feature_train_scaled,labels_train, batch_size=4,verbose=2,epochs=100,callbacks=[stop],validation_split=0.33)

mse,mae = my_model.evaluate(feature_test_scaled,labels_test,verbose=0)

print(mse, mae)

print(history.history.keys())

predicted_values = my_model.predict(feature_test_scaled)
print(r2_score(labels_test,predicted_values))

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax1.plot(history.history['mae'])
ax1.plot(history.history['val_mae'])
ax1.set_title('model mae')
ax1.set_ylabel('MAE')
ax1.set_xlabel('epoch')
ax1.legend(['train','validation'],loc='upper left')

ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('model loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'validation'], loc='upper left')

fig.savefig('static/images/my_plots.png')
