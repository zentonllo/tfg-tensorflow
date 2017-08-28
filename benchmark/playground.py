# -*- coding: utf-8 -*-
"""
Created on Fri May 26 11:03:21 2017

@author: Alberto Terceño

Module that uses deep neural networks to save logs and detailed results of the training process

The code was built up from scratch

"""
import tensorflow as tf
import argparse
from dataset import Dataset
from dnn_multiclass import *
# Uncomment the next line and comment the previous one if you want to use dnn with just one output neuron
#from dnn_binary import *
import sys
import os
from os.path import abspath
from leaky_relu import leaky_relu
from datetime import datetime

# Disable info warnings from TF
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Parser FLAGS
FLAGS = None

# Change DEFAULT_ROOT_LOGDIR for the default log-dir
NOW = datetime.now().strftime("%Y-%m-%d--%Hh%Mm%Ss")
DEFAULT_ROOT_LOGDIR = '/tmp'
DEFAULT_LOG_DIR = "{}/playground-run-{}".format(DEFAULT_ROOT_LOGDIR, NOW)

# Default value for Momentum optimizer
MOMENTUM_PARAM = 0.9

# Default values for the FTRL optimizer
L1_PARAM = 0.0
L2_PARAM = 0.0

def print_hidden_layers(hidden_layers):
    """Helper function to print the number of neurons in the hidden layers"""
    if hidden_layers is None:
        print("Hidden Layers: None (Logistic regression is performed)")
    else:
        i = 1
        for hl in hidden_layers:
            print("Hidden Layer", i,  ":", hl, "neurons")
            i += 1

def print_parameters(n_inputs, n_outputs, normalizer_params):
    """Helper function to print model hyperparameters."""
    print("Model hyperparameters (Binary classification problem)", "\n") 
    print("Input variables:", n_inputs)
    print_hidden_layers(FLAGS.hidden_layers)
    print("Output variables:", n_outputs, "\n")    
    print("Learning Rate:", FLAGS.learning_rate)
    print("Activation Function:", FLAGS.activation_function)
    print("Dropout Keep Probability:", FLAGS.dropout)
    print("Batch size:", FLAGS.batch_size)
    print("Regularization:", FLAGS.regularization)
    print("Regularization parameter (beta):", FLAGS.reg_param)
    
    batch_normalization = FLAGS.batch_norm
    
    if batch_normalization:
        bn = 'Yes'
    else:
        bn = 'No'
        
    print("Batch normalization:", bn)
    if batch_normalization:
        print("Batch normalization parameters:", normalizer_params)
    
    print("Optimizer:", FLAGS.optimizer, "\n")

def parse_act_function():
    """Function which parses the activation function."""
    fun = FLAGS.activation_function
    tf_fun = None
    
    if fun is 'elu':
        tf_fun = tf.nn.elu
    elif fun is 'leaky_relu':
        tf_fun = leaky_relu
    elif fun is 'relu':
        tf_fun = tf.nn.relu
    elif fun is 'sigmoid':
        tf_fun = tf.nn.sigmoid
    elif fun is 'tanh':
        tf_fun = tf.nn.tanh
    elif fun is 'identity':
        tf_fun = tf.nn.identity
    
    return tf_fun


def parse_optimizer():
    """Function which parses the optimization for gradient descent."""
    opt = FLAGS.optimizer
    learning_rate = FLAGS.learning_rate
    
    tf_opt = None
    
    if opt is 'adam':
        tf_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, name='optimizer')
    elif opt is 'adagrad':
        tf_opt = tf.train.AdagradOptimizer(learning_rate=learning_rate, name='optimizer')
    elif opt is 'adadelta':
        tf_opt = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, name='optimizer')
    elif opt is 'ftrl':
        tf_opt = tf.train.FtrlOptimizer(learning_rate=learning_rate,l1_regularization_strength=L1_PARAM, l2_regularization_strength=L2_PARAM, name='optimizer')
    elif opt is 'rms_prop':
        tf_opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate, name='optimizer')
    elif opt is 'momentum':
        tf_opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=MOMENTUM_PARAM, name='optimizer')
    elif opt is 'grad_descent':
        tf_opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name='optimizer')
    
    return tf_opt

def parse_regularizer():
    """Function that parses the regularization methods."""
    reg = FLAGS.regularization
    beta = FLAGS.reg_param
    
    tf_reg = None
    
    if reg is None: 
        return None
    if reg is 'L1': 
        tf_reg = tf.contrib.layers.l1_regularizer(scale=beta, scope=None)
    elif reg is 'L2':
        tf_reg = tf.contrib.layers.l2_regularizer(scale=beta, scope=None)
    
    return tf_reg
    

def parse_normalizer():
    """Function which parses parameters for batch normalization.

    If batch normalization is used in the first layer then the input data is normalized
    Batch normalization can be used with higher learning rates

    """

    if FLAGS.batch_norm:
        normalizer_fn=tf.contrib.layers.batch_norm
    else:
        return None, None
    
    scale_term = None
    if FLAGS.activation_function is 'relu':
        scale_term = False
    else:
        scale_term = True
    
    normalizer_params = {
    'is_training': None,
    # 0.9, 0.99, 0.999 or 0.9999 ...
    # According to TF performance guide: lower it if training is ok and validation/test is performing worse
    # A.Geron suggest to try higher values for large datasets and small batch sizes 
    'decay': 0.9,
    'updates_collections': None,
    # If we don't use activation functions --> scale:true
    'scale': scale_term,
    # The 'fused parameter' allows better performance according to the TF performance guide
    'fused': True
    
    # Try zero_debias_moving_mean=True for improved stability
    # 'zero_debias_moving_mean':True


    }
    
    return normalizer_fn, normalizer_params
    
    

def main(_):
    
    log_dir = FLAGS.log_dir
    
    log_dir = abspath(log_dir)
    

    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)
    
    # Default paths for checkpoints and files generated
    M_FOLDER = abspath(log_dir + '/model')
    TR_FOLDER = abspath(log_dir + '/training')
    
    M_PATH = abspath(M_FOLDER + '/DNN.ckpt')
    TR_PATH = abspath(TR_FOLDER + '/DNN_tr.ckpt')
    
    ROC_PATH = abspath(log_dir + '/roc.png')
    CM_PATH = abspath(log_dir + '/cm.png')
    CM_PATH_NORM = abspath(log_dir + '/cm_norm.png')
    

    os.makedirs(M_FOLDER, exist_ok=True)
    os.makedirs(TR_FOLDER, exist_ok=True)
    # Equivalent to:
    # tf.gfile.MakeDirs(M_FOLDER)
    # tf.gfile.MakeDirs(TR_FOLDER)
    
     
    OUTPUT_FILE = os.path.abspath(log_dir+"/log.txt")
    # Redirect standard output to the log file
    sys.stdout = open(OUTPUT_FILE, "w")
    
    # El path va sin la extensión. El módulo Dataset se encarga de adjuntar la extensión
    # Recall that the file path doesn't have the extension. The Dataset class handles this.
    dataset_path = FLAGS.dataset_file
    
    
    # Data ingestion stage
    print("--------------------- (1) Starting to load dataset ---------------------","\n")
    
    dataset = Dataset(path = dataset_path, train_percentage = 0.8, test_percentage = 0.1 )
    x_test = dataset.x_test
    y_test = dataset.y_test
    print("Number of samples: ", dataset._num_examples)
    print("Number of features: ", dataset._num_features)
    
    print("--------------------- Dataset", dataset_path, "succesfully loaded ---------------------","\n")
    
    
    # We start to parse the hyperparameters 
    n_inputs  = dataset._num_features
    
    n_outputs = dataset._num_classes
    
    # Parsing hidden layers
    intermediate_layers = []
    if FLAGS.hidden_layers is not None:
        intermediate_layers = FLAGS.hidden_layers
    hidden_list = [n_inputs] + intermediate_layers + [n_outputs]
    
    # Parsing activation functions
    activation_function = parse_act_function()
  
    # (1 - keep_prob) is the dropout rate
    keep_prob = FLAGS.dropout
    
    nb_epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size
    regularizer = parse_regularizer() 
    normalizer_fn, normalizer_params = parse_normalizer()
    optimizer = parse_optimizer()
    
    # Print parameters used in the model
    print_parameters(n_inputs, n_outputs, normalizer_params)
    
    print("--------------------- (2) Starting to create the computational graph  ---------------------","\n")
    
    dnn = DNN(log_dir = log_dir,
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
    
    dnn.save_cm(x_test, y_test, model_path=M_PATH, cm_path=CM_PATH_NORM, classes=['Normal transaction','Fraudulent transaction'],normalize=True)
    dnn.save_cm(x_test, y_test, model_path=M_PATH, cm_path=CM_PATH, classes=['Normal transaction','Fraudulent transaction'], normalize=False)
    
    print("--------------------- Confusion matrix saved ---------------------","\n")
    
    
    sys.stdout = sys.__stdout__
    





if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--dataset_file',
                      required=True,
                      type=str,
                      help='Path for the dataset. Do not include the .csv or .npy extension! All the csv columns must be numeric. Label column in the csv is the last one.')
  parser.add_argument('--hidden_layers',
                      type=int,
                      default=None,
                      nargs='*',
                      help='Number of neurons in the hidden layers. Use None if logistic regression wants to be performed.')
  parser.add_argument('--epochs', type=int, default=200,
                      help='Number of epochs to train the model.')
  parser.add_argument('--batch_size', type=int, default=500,
                      help='Batch size used during training.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate.')
  parser.add_argument('--dropout', type=float, default=None,
                      help='Keep probability for training dropout, ie 1-dropout_rate. Use None to avoid using dropout')
  parser.add_argument('--activation_function', type=str, default='elu',
                      help='Activation function to use in the hidden layers: elu, relu, leaky_relu, sigmoid, tanh, identity.')
  parser.add_argument('--optimizer', type=str, default='adam',
                      help='Optimization method to use during training: adam, adagrad, rms_prop, ftrl, adadelta, momentum, grad_descent.')
  parser.add_argument('--batch_norm', dest='batch_norm', action='store_true', default=True,
                      help='Indicate whether to use batch normalization.')
  parser.add_argument('--no_batch_norm', dest='batch_norm', action='store_false', default=False,
                      help='Indicate whether to avoid batch normalization.')
  parser.add_argument('--regularization', type=str, default=None,
                      help='Indicate whether to use L1 or L2 regularization. Use None to avoid regularization')
  parser.add_argument('--reg_param', type=float, default=None,
                      help='Beta parameter for the regularization.')
  parser.add_argument('--log_dir', type=str, default=DEFAULT_LOG_DIR,
                      help='Log directory to store images and TensorBoard summaries')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

