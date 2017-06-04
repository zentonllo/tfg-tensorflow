# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import sys
import tensorflow.contrib.keras as keras
from keras.regularizers import l1,l2
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.constraints import max_norm
from keras.optimizers import RMSprop, Adam
from keras.layers.normalization import BatchNormalization
from keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint, EarlyStopping
from numpy import genfromtxt
from sklearn.metrics import roc_auc_score
from dataset import Dataset
from keras_interval_evaluation import IntervalEvaluation
from datetime import datetime
from os.path import abspath
import os


# Disable info warnings from TF
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

NOW = datetime.now().strftime("%Y-%m-%d--%Hh%Mm%Ss")
ROOT_LOGDIR = 'B:/tmp/logs_keras'
LOG_DIR = abspath('{}/run-{}'.format(ROOT_LOGDIR, NOW))
CSV_LOG = abspath(LOG_DIR + '/training.log')
TXT_LOG = abspath(LOG_DIR + '/results.txt')
CKPT = abspath(LOG_DIR + '/ckpt.hdf5')
OUTPUT_FILE = abspath(TXT_LOG)
MODEL = abspath(LOG_DIR + '/model.h5')

if tf.gfile.Exists(LOG_DIR):
    tf.gfile.DeleteRecursively(LOG_DIR)
tf.gfile.MakeDirs(LOG_DIR)


# Parameters for early stopping (increase them when using auc scores)
DELTA = 1e-6
PATIENCE = 200

# Auc callback interval
AUCS_INTERVAL = 1

# Hyperparameters
batch_size = 500
epochs = 200
dropout_rate = 0.5
beta = 0

dataset = Dataset(path='creditcards', train_percentage=0.7, test_percentage=0.2)
x_train = dataset.x_train
y_train = dataset.y_train
x_val = dataset.x_val
y_val = dataset.y_val
x_test = dataset.x_test
y_test = dataset.y_test

input_dim = dataset._num_features
num_classes = 2

"""
Regularizacion:
model.add(Dense(100, activation='relu', 
                kernel_regularizer=l2(beta), 
                activity_regularizer=l1(beta),
                input_shape=(input_dim,)))

También se pueden inicializar pesos de diferentes maneras
kernel_initializer

Restricciones de pesos (añadir como parámetro de Dense)
kernel_constraint=max_norm(2.)

"""
model = Sequential()
model.add(Dense(5,input_shape=(input_dim,), init='he_normal'))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Dropout(dropout_rate))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])
# CALLBACKS
# Minimizing and stopping until val_loss reaches a minimum
early_stopping = EarlyStopping(min_delta = DELTA, patience = PATIENCE )
ckpt = ModelCheckpoint(filepath = CKPT, save_best_only = True)
ival = IntervalEvaluation(validation_data = (x_val, y_val), interval = AUCS_INTERVAL)
tb = TensorBoard(log_dir = LOG_DIR,histogram_freq = 1, 
                 write_graph = True, write_images = False)
csv_logger = CSVLogger(CSV_LOG)

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_val, y_val),
                    callbacks=[ival, csv_logger, tb, ckpt, early_stopping])

model.save(MODEL)

# Habría que hacer model = load_model(MODEL_PATH) y luego model.load_wwights(CKPT_PATH)
# También podemos reconstruir el modelo con model = model_from_json(json_string)
model.load_weights(CKPT)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1], "\n")



y_pred = model.predict_proba(x_test, verbose = 0)
y_score = y_pred[:,1]
auc = roc_auc_score(y_true=y_test, y_score=y_score)
auc *=100
print("Test AUC:", auc)

# Write the Validation AUCS and another important results in a file
sys.stdout = open(OUTPUT_FILE, "w")
json_string = model.to_json() 
print("Network structure (json format)", "\n")
print(json_string, "\n")
print("Hyperparameters", "\n")
print("Batch size:", batch_size)
print("Epochs:", epochs)
print("Dropout rate:", dropout_rate, "\n")
model.summary()
print("Validation AUCs during training", "\n")
for i in range(len(ival.aucs)):
    print("\t","Epoch", str(i), "- val_auc:", ival.aucs[i], " - loss:", ival.losses[i])

print('\n','Test loss:', score[0])
print('Test accuracy:', score[1], "\n")
print('Test AUC:', auc)
sys.stdout = sys.__stdout__