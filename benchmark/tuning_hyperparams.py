# -*- coding: utf-8 -*-
"""
Created on Fri May  5 11:26:14 2017

@author: Alberto Terceño



######## Hyperparameter tuning  - Instructions ##############################

 1 - Choose a dataset (just the name of the csv or npy file)
 2 - Choose the hyperparameters grid to test. Fill up the different hyperparams lists in
    order to test all the combinations.

#############################################################################


"""



import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from dnn_multiclass import DNN
# Uncomment the next line and comment the previous one if you want to use dnn with just one output neuron
# from dnn_binary import *
from dataset import Dataset
import sys
import time
from datetime import datetime
import os

# Disable warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def print_execution_time(start, end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return ("{:0>2}:{:0>2}:{:0>2}".format(int(hours),int(minutes),int(seconds)))


def print_parameters(lr,hl,af,kp,bs,reg,bn,opt):
    print("Hyperparameters", "\n")
    print("Learning rate:", lr)
    print("Hidden layers (Including input and output layers):", hl)
    print("Activation function:", af)
    print("Dropout keep prob rate (None if not used):", kp)
    print("Batch size:", bs)
    print("Regularizer (None if not used):", reg)
    print("Batch normalization (Yes = 1, No = 0):", bn)
    print("Optimizer:", opt, "\n")

NOW = datetime.now().strftime("%Y-%m-%d--%Hh%Mm%Ss")
DEFAULT_ROOT_LOGDIR = '/tmp'
DEFAULT_LOG_DIR = "{}/hyptuning-run-{}".format(DEFAULT_ROOT_LOGDIR, NOW)
OUTPUT_FILE = DEFAULT_LOG_DIR + "/tuning_results.txt"

if tf.gfile.Exists(DEFAULT_LOG_DIR):
    tf.gfile.DeleteRecursively(DEFAULT_LOG_DIR)
tf.gfile.MakeDirs(DEFAULT_LOG_DIR)
    

sys.stdout = open(OUTPUT_FILE, "w")



######################################## Choose a dataset path #####################################################
####################################################################################################################

dataset_name = 'creditcards'

###################################################################################################################
###################################################################################################################


######################################## Choose lists of hyperparameters to test ##################################
###################################################################################################################

nb_epochs = 30

# Use np.linspace() to get several values between a minimum and a maximum
learning_rate_list = [0.1,0.5]
#learning_rate_list = [0.001]

activation_function_list = [tf.nn.elu]
#activation_function_list = [tf.nn.relu, tf.nn.elu]

keep_prob_list = [None]
#keep_prob_list = [0.2,0.4,0.5,0.7]

batch_size_list = [100]

regularizer_list = [None]
#beta = 0.001
#regularizer_list = [None,tf.contrib.layers.l1_regularizer(scale=beta, scope=None),  tf.contrib.layers.l2_regularizer(scale=beta, scope=None)]
'''
normalizer_fn_list = [batch_norm]
normalizer_params_list = [{
        'is_training': None,
        'decay': 0.9,
        'updates_collections': None,
        'scale': True,
    }]
'''
normalizer_fn_list = [None, batch_norm]
normalizer_params_list = [None, {
        'is_training': None,
        'decay': 0.9,
        'updates_collections': None,
        'scale': True,
    }]
#'''

optimizers_list = [tf.train.AdamOptimizer]
#optimizers_list = [tf.train.AdamOptimizer, tf.train.RMSPropOptimizer, tf.train.AdadeltaOptimizer,
#                   tf.train.AdagradOptimizer, tf.train.RMSPropOptimizer]




dataset = Dataset(path = dataset_name, train_percentage = 0.8, test_percentage = 0.1 )
x_test = dataset.x_test
y_test = dataset.y_test

n_inputs = dataset._num_features
n_outputs = dataset._num_classes
#hidden_lists= [[n_inputs, 10, 5, n_outputs], [n_inputs, 10, n_outputs], [n_inputs, 3, n_outputs]]
hidden_lists= [[n_inputs, 10, n_outputs], [n_inputs, 3, n_outputs]]


####################################################################################################################
####################################################################################################################

best_auc = 0
it = 0

start_time = time.time()

for learning_rate in learning_rate_list:
    for hidden_list in hidden_lists:
        for activation_function in activation_function_list:
            for keep_prob in keep_prob_list:
                for batch_size in batch_size_list:
                    for regularizer in regularizer_list:
                        for i in range(len(normalizer_fn_list)):
                            for optimizer in optimizers_list:
                                
                                log_dir_model = DEFAULT_LOG_DIR + "/model" + str(it) 
                                os.makedirs(log_dir_model, exist_ok=True)
                                os.makedirs(log_dir_model + "/model_tuning", exist_ok=True)
                                os.makedirs(log_dir_model + "/train_tuning", exist_ok=True)
                                
                                start_time_model = time.time()
                                
                                dnn = DNN(log_dir = log_dir_model,
                                          hidden_list=hidden_list,
                                          activation_function = activation_function,
                                          keep_prob = keep_prob,
                                          regularizer = regularizer,
                                          normalizer_fn = normalizer_fn_list[i],
                                          normalizer_params = normalizer_params_list[i],
                                          optimizer = optimizer(learning_rate, name='optimizer'))
                                
                                model_path = log_dir_model + "/model_tuning/DNN" + str(it) + ".ckpt"
                                train_path = log_dir_model + "/train_tuning/DNN" + str(it) + ".ckpt"
                                
                                print("--------------------- Model", it+1, " ------------------------")
                                print("-------------------------------------------------------", "\n" )
                                print_parameters(learning_rate,hidden_list,activation_function,keep_prob,batch_size,regularizer,i,optimizer)
                                print("------- Starting to train model", it+1, " -------", "\n" )
                                dnn.train(dataset=dataset, nb_epochs=nb_epochs, 
                                          batch_size=batch_size, 
                                          model_path=model_path, 
                                          train_path=train_path, 
                                          silent_mode=True)
                                print("----- Training for model", it+1, "finished -----", "\n" )
                                
                                print("------- Test for model", it+1, " -------", "\n" )
                                dnn.test(x_test=x_test, y_test=y_test, model_path=model_path)
                                print("------- Test for model", it+1, " finished -------", "\n" )
                                print("---------------------------------------------------------")
                                print("---------------------------------------------------------", "\n")
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
                                print("Execution time for model", it, ":", print_execution_time(start_time_model, time.time()))
                                



print("\n","\n","----------------------------------------------------")
print("----------------------------------------------------")
print("----------------------------------------------------", "\n")
print("Execution time for ", it, "models: ", print_execution_time(start_time, time.time()),"\n")


print("Best result (AUC):", best_auc)
print("Number of epochs: ", nb_epochs,"\n")
print_parameters(lr,hl,af,kp,bs,reg,bn,opt)
print("----------------------------------------------------")
print("----------------------------------------------------")
print("----------------------------------------------------")

sys.stdout = sys.__stdout__
