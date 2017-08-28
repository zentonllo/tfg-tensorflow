# -*- coding: utf-8 -*-
"""
Module testing some Keras features. This model was coded from scratch and it inspired 
a couple of Datalab Notebooks (Keras-GCS.ipynb adn Keras-BQ.ipynb)

@author: Alberto Terceño
"""

from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import sys
# Import keras from the core TensorFlow library
# This allows us to avoid installing Keras as a separate Python package
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
ROOT_LOGDIR = '/tmp/keras_logs_tests'
LOG_DIR = abspath('{}/run-{}'.format(ROOT_LOGDIR, NOW))
CSV_LOG = abspath(LOG_DIR + '/training.log')
TXT_LOG = abspath(LOG_DIR + '/results.txt')
CKPT = abspath(LOG_DIR + '/ckpt.hdf5')
OUTPUT_FILE = abspath(TXT_LOG)
MODEL = abspath(LOG_DIR + '/model.h5')

if tf.gfile.Exists(LOG_DIR):
    tf.gfile.DeleteRecursively(LOG_DIR)
tf.gfile.MakeDirs(LOG_DIR)


# Parameters for early stopping (increase them when minimizing cost functions with several plateaus)
DELTA = 1e-6
PATIENCE = 200

# Auc callback interval
AUCS_INTERVAL = 1

# Hyperparameters
batch_size = 500
epochs = 10
dropout_rate = 0.5
beta = 0

dataset = Dataset(path='creditcard', train_percentage=0.8, test_percentage=0.1)
x_train = dataset.x_train
y_train = dataset.y_train
x_val = dataset.x_val
y_val = dataset.y_val
x_test = dataset.x_test
y_test = dataset.y_test

input_dim = dataset._num_features
num_classes = dataset._num_classes

"""
How to add L1, L2 regularization :

model.add(Dense(100, activation='relu', 
                kernel_regularizer=l2(beta), 
                activity_regularizer=l1(beta),
                input_shape=(input_dim,)))

Use 'kernel_initializer' to initialize weights in different fashions

Add kernel_constraint=max_norm(2.) for restrictions in weight restrictions

"""
model = Sequential()
# Using a single hidden layer with 5 neurons and He initialization
model.add(Dense(5,input_shape=(input_dim,), kernel_initializer='he_normal'))
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

# Save the Keras model as a h5 file
model.save(MODEL)

# We can reuse the model using model = load_model(MODEL_PATH) and then model.load_weights(CKPT_PATH)
# We can also use model = model_from_json(json_string) using the saved JSON in the log file

# Restore model from checkpoint ('best' trained model)
model.load_weights(CKPT)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1], "\n")


# Compute AUC from model predictions
y_pred = model.predict_proba(x_test, verbose = 0)
y_score = y_pred[:,1]
auc = roc_auc_score(y_true=y_test, y_score=y_score)
auc *=100
print("Test AUC:", auc)

# Write the validation AUCS and more detailed results in a file
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
    print("Epoch", str(i), "- val_auc:", ival.aucs[i], " - loss:", ival.losses[i])

print('\n','Test loss:', score[0])
print('Test accuracy:', score[1], "\n")
print('Test AUC:', auc)
sys.stdout = sys.__stdout__