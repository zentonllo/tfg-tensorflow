# -*- coding: utf-8 -*-
"""
Created on Fri May 26 11:03:21 2017

@author: Alberto Terceño
"""
import tensorflow as tf
import time
import numpy as np
import argparse
from dataset import Dataset

import sys
import os
import matplotlib.pyplot as plt
import itertools

from tensorflow.contrib.layers import fully_connected, batch_norm, dropout
from tensorflow.contrib.framework import arg_scope
from leaky_relu import leaky_relu

from datetime import datetime
from sklearn.metrics import auc, roc_auc_score, roc_curve, confusion_matrix

# Disable info warnings from TF
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

FLAGS = None

NOW = datetime.now().strftime("%Y-%m-%d--%Hh%Mm%Ss")
DEFAULT_ROOT_LOGDIR = 'C:/tmp'
DEFAULT_LOG_DIR = "{}/run-{}".format(DEFAULT_ROOT_LOGDIR, NOW)

"""
n_inputs: Número de variables de entrada
n_outputs: Número de clases objetivo (2 problemas clasificación binaria, >2 problemas clasificación multiclase)

learning_rate: Tasa de aprendizaje (suele ser 0.001)
hidden_list: Lista de capas ocultas (incluidas neuronas input y outpt, ej: [n_input, 300, 300, n_output])
activation_function: Funciones de activación (ej: tf.nn.relu, tf.nn.elu, tf.nn.sigmoid, tf.nn.tanh, tf.nn.identity)
keep_prob: Keep probability para el Dropout (suele ser 0.5)
regularizer: Regularizer a usar (ej: tf.contrib.layers.l1_regularizer, tf.contrib.layers.l2_regularizer)

normalizer_fn: función de normalización (None o batch_norm para realizar batch normalization)
# "batch_norm_params", parámetros para batch normalization
normalizer_params = {
    'is_training': is_training,
    'decay': 0.9,
    'updates_collections': None,
    'scale': True,
}

optimizer: = tf.train.AdamOptimizer, tf.train.RMSPropOptimizer, tf.train.AdadeltaOptimizer, tf.train.AdagradOptimizer, tf.train.MomentumOptimizer (este requiere cambios)
"""

# Sustituir todo por FLAGS. ...

# BN se incluiría con un booleano
# BN vendría sin parámetros por defecto pero podremos incluirlos
def print_parameters(**kwargs):
    
    print("Model hyperparameters", "\n") 
    print("Learning Rate:", kwargs['learning_rate'])
    print("Hidden Layers:", kwargs['hidden_layers'])
    print("Activation Function:", kwargs['activation_function'])
    print("Dropout Keep Probability:", kwargs['keep_prob'])
    print("Batch size:", kwargs['batch_size'])
    print("Regularization:", kwargs['regularization'])
    
    batch_normalization = kwargs['batch_normalization']
    
    if batch_normalization:
        bn = 'Si'
    else:
        bn = 'No'
        
    print("Batch normalization:", bn)
    if batch_normalization:
        print("Batch normalization parameters:", kwargs['bn_params'])
    
    print("Optimizer:", kwargs['optimizer'], "\n")



def main(_):
    
    log_dir = FLAGS.log_dir
    
    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)
    
    # Checkpoints default paths
    M_FOLDER = log_dir + '/model'
    TR_FOLDER = log_dir + '/training'
    
    M_PATH = M_FOLDER + '/DNN.ckpt'
    TR_PATH = TR_FOLDER + '/DNN_tr.ckpt'
    
    ROC_PATH = log_dir + '/roc.png'
    CM_PATH = log_dir + '/cm.png'
    CM_PATH_NORM = log_dir + '/cm_norm.png'
    
    os.makedirs(M_FOLDER, exist_ok=True)
    os.makedirs(TR_FOLDER, exist_ok=True)
    
     
    OUTPUT_FILE = log_dir+"/log.txt"
    sys.stdout = open(OUTPUT_FILE, "w")
    
    dataset_path = FLAGS.train_files
    
    
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
    # dataset_path = 'data_regf_it17'
  
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
    
    
    nb_epochs = 400
    #batch_size = 100000
    #batch_size = 20000
    # batch_size = 500
    batch_size = 20
    
    #regularizer = tf.contrib.layers.l2_regularizer
    #beta = 0.001
    regularizer = None
    beta = None
    
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
    
    dnn = DNN(log_dir = log_dir,
              hidden_list=hidden_list,
              activation_function = activation_function,
              keep_prob = keep_prob,
              regularizer = regularizer,
              beta = beta,
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
    





if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--train_files',
                      required=True,
                      type=str,
                      help='Path for training csv files', nargs='+')
  parser.add_argument('--epochs', type=int, default=200,
                      help='Number of epochs to train the model.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=None,
                      help='Keep probability for training dropout.')
  parser.add_argument('--batch_norm', type=bool, default=True,
                      help='Indicate whether to use batch normalization.')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')  
  parser.add_argument('--log_dir', type=str, default=DEFAULT_LOG_DIR,
                      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

