# -*- coding: utf-8 -*-
"""
Created on Mon May 29 13:47:37 2017

@author: Alberto Terceño
"""

"""
This is a test playground. The hyperparameters used in the DNN class are harcoded 
The csv used was downloaded from Kaggle: https://www.kaggle.com/dalpozz/creditcardfraud
"""

from datetime import datetime
import os 
import sys
import tensorflow as tf

from importlib import reload
from os.path import abspath
from nn import DNN
#from dnn_binary import *
from dataset import Dataset
from tensorflow.contrib.layers import batch_norm

# Disable info warnings from TF
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Pasar estos datos como parámetros a la clase
NOW = datetime.now().strftime("%Y-%m-%d--%Hh%Mm%Ss")
ROOT_LOGDIR = '/tmp'
LOG_DIR = "{}/run-{}".format(ROOT_LOGDIR, NOW)

LOG_DIR = abspath(LOG_DIR)

# Checkpoints default paths
M_FOLDER = abspath(LOG_DIR + '/model')
TR_FOLDER = abspath(LOG_DIR + '/training')

M_PATH = abspath(M_FOLDER + '/DNN.ckpt')
TR_PATH = abspath(TR_FOLDER + '/DNN_tr.ckpt')

ROC_PATH = abspath(LOG_DIR + '/roc.png')
CM_PATH = abspath(LOG_DIR + '/cm.png')
CM_PATH_NORM = abspath(LOG_DIR + '/cm_norm.png')

os.makedirs(M_FOLDER, exist_ok=True)
os.makedirs(TR_FOLDER, exist_ok=True)

 
OUTPUT_FILE = abspath(LOG_DIR+"/log.txt")
sys.stdout = open(OUTPUT_FILE, "w")


print("--------------------- (1) Starting to load dataset ---------------------","\n")

# Dataset para detección de fraude en transacciones con tarjetas de crédito
# n_inputs = 30 (Time, V1-V28 y Amount)
# n_outputs = 2
dataset_path = os.path.abspath('./creditcards')

  
dataset = Dataset(path = dataset_path, train_percentage = 0.8, test_percentage = 0.1 )
x_test = dataset.x_test
y_test = dataset.y_test

print("--------------------- Dataset", dataset_path, "succesfully loaded ---------------------","\n")

n_inputs  = dataset._num_features
n_outputs = 2

# Hyperparameters 

# Con Batch normalization se pueden usar grandes learning rates
learning_rate = 0.001
hidden_list = [n_inputs, 5, n_outputs]
activation_function = tf.nn.elu

# (1 - keep_prob) es la probabilidad de que una neurona muera en el dropout
keep_prob = 0.5
nb_epochs = 200
batch_size = 500
regularizer = None


# Batch norm en la primera capa asegura normalización de los datos
normalizer_fn=batch_norm
# Batch_norm_params
normalizer_params = {
    'is_training': None,
    # 0.9 o 0.99 o 0.999 o 0.9999 ...
    # Segun performance guide de TF: menor si va bien en training y mej
    # Según A.Geron, aumentar cuando el dataset es grande y los batches pequeños 
    'decay': 0.9,
    'updates_collections': None,
    # Si usamos funciones de activacion que no sean relus --> scale:true
    'scale': True,
    # Aumenta rendimiento según la performance guide de TF
    'fused': True
    
    # Try zero_debias_moving_mean=True for improved stability
    # 'zero_debias_moving_mean':True

}


optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='optimizer')

print("--------------------- (2) Starting to create the computational graph  ---------------------","\n")

dnn = DNN(log_dir = LOG_DIR,
          hidden_list=hidden_list,
          activation_function = activation_function,
          keep_prob = keep_prob,
          regularizer = regularizer,
          normalizer_fn = normalizer_fn,
          normalizer_params = normalizer_params,
          optimizer = optimizer)


print("--------------------- Graph created ---------------------","\n")

print("--------------------- (3) Starting training ---------------------","\n")

dnn.train(dataset=dataset,model_path=M_PATH, train_path=TR_PATH, nb_epochs=nb_epochs, batch_size=batch_size, silent_mode=False)

print("--------------------- Training Finished ---------------------","\n")

print("--------------------- (4) Starting test ---------------------","\n")

dnn.test(x_test=x_test, y_test=y_test, model_path=M_PATH)

print("--------------------- Test Finished ---------------------","\n")


print("--------------------- (5) Saving model ROC curve ---------------------","\n")

dnn.save_roc(x_test, y_test, model_path=M_PATH, roc_path=ROC_PATH)

print("--------------------- ROC curve saved ---------------------","\n")

print("--------------------- (5) Saving confusion matrix ---------------------","\n")

dnn.save_cm(x_test, y_test, model_path=M_PATH, cm_path=CM_PATH_NORM, classes=['User not booking','User booking'],normalize=True)
dnn.save_cm(x_test, y_test, model_path=M_PATH, cm_path=CM_PATH, classes=['User not booking','User booking'], normalize=False)

print("--------------------- Confusion matrix saved ---------------------","\n")


sys.stdout = sys.__stdout__

