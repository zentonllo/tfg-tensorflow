from __future__ import print_function

import numpy as np
import tensorflow.contrib.keras as keras
from keras.regularizers import l1,l2
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.constraints import max_norm
from keras.optimizers import RMSprop, Adam
from numpy import genfromtxt
from sklearn.metrics import roc_auc_score
from dataset import Dataset

import os
# Disable info warnings from TF
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

batch_size = 500
epochs = 20
dropout_rate = 0.7
beta = 0

dataset = Dataset(path='data_regf_it17', train_percentage=0.8)
x_train = dataset.x_train
y_train = dataset.y_train
x_test = dataset.x_test
y_test = dataset.y_test

input_dim = x_test.shape[1]
num_classes = np.unique(y_test).size

"""
Regularizacion:
model.add(Dense(100, activation='relu', 
                kernel_regularizer=l2(beta), 
                activity_regularizer=l1(beta),
                input_shape=(input_dim,)))

También se pueden inicializar pesos de diferentes maneras
kernel_initializer

Restricciones de pesos
kernel_constraint=max_norm(2.)

"""
model = Sequential()
model.add(Dense(5, activation='elu',
                kernel_constraint=max_norm(2.),
                input_shape=(input_dim,)))
model.add(Dropout(dropout_rate))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])



history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

# model = load_model('modelo_dnn.h5')
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('modelo_dnn.h5')


y_pred = model.predict_proba(x_test)
y_score = y_pred[:,1]
auc = roc_auc_score(y_true=y_test, y_score=y_score)
print("\n", "AUC:", auc*100)

