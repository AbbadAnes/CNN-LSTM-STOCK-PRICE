# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
import numpy

# Read recipe inputs
stock_prices_train = dataiku.Dataset("stock_prices_train")
stock_prices_train_df = stock_prices_train.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
stock_prices_train_df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
stock_prices_train_df = stock_prices_train_df.drop(columns=['Date'])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
stock_prices_train_df = (stock_prices_train_df-stock_prices_train_df.mean())/stock_prices_train_df.std()
stock_prices_train_df.tail()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
X = []
y = []

window_size=20
predict_size=1

for i in range(1 , len(stock_prices_train_df) - window_size - predict_size , 1):
    first = np.array(stock_prices_train_df.iloc[i])
    temp = []
    temp2 = []

    for j in range(window_size):
        temp.append(np.array(stock_prices_train_df.iloc[i + j]))

    for k in range(predict_size):
        temp2.append(np.array(stock_prices_train_df.iloc[i + window_size + k]))

    X.append(np.array(temp).reshape(window_size, 17))
    y.append(np.array(temp2).reshape(predict_size, 17))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
index_of_colum_to_keep = 4

y_prepared = []
X_prepared = []

for i in y:
    y_prepared.append(i[0][index_of_colum_to_keep])

X_train = []
temp = []

for j in X:
    for z in j:
        temp.append(z[index_of_colum_to_keep]) 
    X_prepared.append(temp)
    temp = []

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
X_prepared = np.array(X_prepared)
y_prepared = np.array(y_prepared)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
X_prepared.shape, y_prepared.shape

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
test_subset = 1500

X_test = X_prepared[-test_subset:]
X_train = X_prepared[:len(X_prepared)-test_subset]

y_test = y_prepared[-test_subset:]
y_train = y_prepared[:len(y_prepared)-test_subset]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
len(X_train), len(y_train), len(X_test), len(y_test),

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
X_train = np.reshape(X_train, (len(X_train), 20,1))
y_train = np.reshape(y_train, len(y_train))
X_test = np.reshape(X_test, (len(X_test), 20,1))
y_test = np.reshape(y_test, len(y_test))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
nb_epochs = 200

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# For creating model and training
import tensorflow as tf

from tensorflow.python.keras.layers import Conv1D, Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Bidirectional, TimeDistributed, MaxPooling1D, Flatten

model = tf.keras.Sequential()

# CNN layers

model.add(Conv1D(64, 2, activation='relu', input_shape=(20, 1)))

#model.add(Dense(64, activation="relu"))

model.add(MaxPooling1D())

model.add(Conv1D(128, kernel_size=3, activation='relu'))

model.add(MaxPooling1D())

# model.add(Conv1D(64, kernel_size=3, activation='relu'))

# model.add(MaxPooling1D(2))

# model.add(Flatten())

#LSTM layers

#model.add(Bidirectional(LSTM(100, return_sequences=True)))

#model.add(Dropout(0.5))

#model.add(Bidirectional(LSTM(100, return_sequences=False)))

#model.add(Dropout(0.5))

model.add(Flatten())

#Final layers

model.add(Dense(units=1, activation='linear'))

model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])

#history = model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=40,batch_size=None, verbose=1, shuffle =True)

history = model.fit(X_train, y_train, epochs=nb_epochs,batch_size=None, verbose=1, shuffle =True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
model.summary()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
acc = model.evaluate(X_train, y_train)
print("Loss:", acc[0], " Accuracy:", acc[1])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
y_predict = model.predict(X_test)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from sklearn.metrics import r2_score

r2_score(y_test, y_predict.flatten())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
models = dataiku.Folder("IXfb24Bs")
models_info = models.get_info()
models_metrics = dataiku.Folder("4cDmNBRV")
models_metrics_info = models_metrics.get_info()