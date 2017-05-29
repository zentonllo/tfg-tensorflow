# -*- coding: utf-8 -*-
"""
Created on Mon May 29 13:47:37 2017

@author: Alberto Terceño
"""

"""This is the original playground, hardcoding the hyperparameters to use in the DNN class"""

import datetime
import os 
import sys
import tensorflow as tf

from nn import DNN
from dataset import Dataset
from tensorflow.contrib.layers import batch_norm

# Disable info warnings from TF
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Pasar estos datos como parámetros a la clase
NOW = datetime.now().strftime("%Y-%m-%d--%Hh%Mm%Ss")
ROOT_LOGDIR = 'C:/tmp'
LOG_DIR = "{}/run-{}".format(ROOT_LOGDIR, NOW)


# Checkpoints default paths
M_FOLDER = LOG_DIR + '/model'
TR_FOLDER = LOG_DIR + '/training'

M_PATH = M_FOLDER + '/DNN.ckpt'
TR_PATH = TR_FOLDER + '/DNN_tr.ckpt'

ROC_PATH = LOG_DIR + '/roc.png'
CM_PATH = LOG_DIR + '/cm.png'
CM_PATH_NORM = LOG_DIR + '/cm_norm.png'

os.makedirs(M_FOLDER, exist_ok=True)
os.makedirs(TR_FOLDER, exist_ok=True)

 
OUTPUT_FILE = LOG_DIR+"/log.txt"
sys.stdout = open(OUTPUT_FILE, "w")


# Carga del dataset 
# En R hacemos previamente: write.table(MyData, file = "MyData.csv",row.names=FALSE, na="",col.names=FALSE, sep=",")
print("--------------------- (1) Starting to load dataset ---------------------","\n")

#OJO Dataset 5M filas! (No balanceado)
# n_inputs = 65
# n_outputs = 2
# dataset_path = 'validation_it17'
 
#Introducir aquí dataset balanceado de gran tamaño
# .....

# Dataset de iteración 17 (260k filas)
# n_inputs = 65
# n_outputs = 2
dataset_path = 'data_regf_it17'
  
# Dataset con todas las variables (260k filas)
# n_inputs = 155
# n_outputs = 2
# dataset_path = 'data_regf_completo'

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
#activation_function = leaky_relu

# (1 - keep_prob) es la probabilidad de que una neurona muera en el dropout
keep_prob = 0.5
#keep_prob = None


nb_epochs = 200
#batch_size = 100000
#batch_size = 20000
batch_size = 500
#batch_size = 20

# beta = 0.001
#regularizer = tf.contrib.layers.l2_regularizer(scale=beta, scope=None)
regularizer = None

"""
normalizer_fn = None
normalizer_params = None
"""
# Batch norm en la primera capa asegura normalización de los datos
normalizer_fn=batch_norm
# "batch_norm_params"
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
# optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate, l1_regularization_strength=0.0, l2_regularization_strength=0.0, name='optimizer')
#optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, name='optimizer')
#optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, name='optimizer')

print("--------------------- (2) Starting to create the computational graph  ---------------------","\n")

dnn = DNN(log_dir = LOG_DIR,
          hidden_list=hidden_list,
          activation_function = activation_function,
          keep_prob = keep_prob,
          regularizer = regularizer,
          normalizer_fn = normalizer_fn,
          normalizer_params = normalizer_params,
          optimizer = optimizer)

# Probablemente haya que usar FLAGS para parsear y para pasar parámetros a la función
# print_parameters(...)


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

