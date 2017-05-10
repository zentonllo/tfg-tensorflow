# -*- coding: utf-8 -*-
"""
Created on Fri May  5 11:26:14 2017

@author: Alberto Terceño
"""

import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from nn import DNN
from dataset import Dataset
import sys

# Desactivamos warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#learning_rate = 0.01
#hidden_list = [n_inputs, 100, 100, n_outputs]
#activation_function = tf.nn.relu
#keep_prob = 0.4
##keep_prob = None
#
#
#nb_epochs = 100
#batch_size = 500
#
##regularizer = tf.contrib.layers.l2_regularizer
##beta = 0.001
#regularizer = None
#beta = None
#
#"""
#normalizer_fn = None
#normalizer_params = None
#"""
#normalizer_fn=batch_norm
## "batch_norm_params"
#normalizer_params = {
#    'is_training': None,
#    'decay': 0.9,
#    'updates_collections': None,
#    'scale': True,
#}
#
#
#optimizer = tf.train.AdamOptimizer
##optimizer = tf.train.RMSPropOptimizer

def print_parameters(lr,hl,af,kp,bs,reg,bn,opt):
    print("Tasa de aprendizaje:", lr, "\n")
    print("Lista de capas ocultas:", hl, "\n")
    print("Función de activación:", af, "\n")
    print("Probabilidad dropout:", kp, "\n")
    print("Batch size:", bs, "\n")
    print("Regularizer:", reg, "\n")
    print("Batch normalization (i=0 No, i=1 Si):", bn, "\n")
    print("Optimizer:", opt, "\n")


OUTPUT_FILE = "tuning_results2.txt"

sys.stdout = open(OUTPUT_FILE, "w")

n_inputs = 65
n_outputs = 2

learning_rate_list = [0.001]
hidden_lists= [[n_inputs, 1, n_outputs], [n_inputs, 2, n_outputs], [n_inputs, 3, n_outputs]]
activation_function_list = [tf.nn.relu, tf.nn.elu]
#keep_prob_list = [0.2,0.4,0.5,0.7]
keep_prob_list = [0.4, 0.7]

nb_epochs = 200
batch_size_list = [100,500]
#regularizer_list = [None, tf.contrib.layers.l2_regularizer]
regularizer_list = [None]
beta = 0

normalizer_fn_list = [None, batch_norm]
normalizer_params_list = [None, {
        'is_training': None,
        'decay': 0.9,
        'updates_collections': None,
        'scale': True,
    }]

#optimizers_list = [tf.train.AdamOptimizer, tf.train.RMSPropOptimizer, tf.train.AdadeltaOptimizer,
#                   tf.train.AdagradOptimizer, tf.train.RMSPropOptimizer]
optimizers_list = [tf.train.AdamOptimizer]


# En R hacemos previamente: write.table(MyData, file = "MyData.csv",row.names=FALSE, na="",col.names=FALSE, sep=",")
dataset = Dataset(path = 'data_regf_it17.csv', train_percentage = 0.8 )
x_test = dataset.x_test
y_test = dataset.y_test

best_auc = 0
it = 0

for learning_rate in learning_rate_list:
    for hidden_list in hidden_lists:
        for activation_function in activation_function_list:
            for keep_prob in keep_prob_list:
                for batch_size in batch_size_list:
                    for regularizer in regularizer_list:
                        for i in range(2):
                            for optimizer in optimizers_list:
                                
                                dnn = DNN(learning_rate=learning_rate,
                                          hidden_list=hidden_list,
                                          activation_function = activation_function,
                                          keep_prob = keep_prob,
                                          regularizer = regularizer,
                                          beta = beta,
                                          normalizer_fn = normalizer_fn_list[i],
                                          normalizer_params = normalizer_params_list[i],
                                          optimizer = optimizer)
                                
                                model_path = "./model_tuning/DNN" + str(it) + ".ckpt"
                                train_path = "./train_tuning/DNN" + str(it) + ".ckpt"
                                
                                print(" --------- Modelo", it+1, " ---------", "\n" )
                                print_parameters(learning_rate,hidden_list,activation_function,keep_prob,batch_size,regularizer,i,optimizer)
                                dnn.train(dataset=dataset, nb_epochs=nb_epochs, 
                                          batch_size=batch_size, 
                                          model_path=model_path, 
                                          train_path=train_path, 
                                          silent_mode=True)
                                
                                dnn.test(x_test=x_test, y_test=y_test, model_path=model_path)
                                
                                print("  -----------------------------------", "\n")
                                curr_auc = dnn.auc_roc(dataset.x_test, dataset.y_test, model_path=model_path)
                                if curr_auc > best_auc:
                                    best_auc = curr_auc
                                    lr = learning_rate
                                    hl = hidden_list
                                    af = activation_function
                                    kp = keep_prob
                                    bs = batch_size
                                    reg = regularizer
                                    bn = i
                                    opt = optimizer
                                it += 1


print("Mejor resultado (AUC):", best_auc, "\n")
print_parameters(lr,hl,af,kp,bs,reg,bn,opt)


sys.stdout = sys.__stdout__
