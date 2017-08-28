'''
A logistic regression learning algorithm example using TensorFlow library and the CCF dataset.

Code obtained from Aymeric Damien (Logistic regression on MNIST dataset)
Project: https://github.com/aymericdamien/TensorFlow-Examples/
Source code: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/logistic_regression.py
'''

import tensorflow as tf
import numpy as np
import time
from dataset import Dataset

# Turn off TF warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from sklearn.metrics import roc_auc_score


def print_time_training(start, end):
	"""Helper function to print execution times properly formatted."""
	hours, rem = divmod(end-start, 3600)
	minutes, seconds = divmod(rem, 60)
	print("Training time:","{:0>2}:{:0>2}:{:0>2}".format(int(hours),int(minutes),int(seconds)))


# Load the Dataset using the data ingestion module 
dataset = Dataset(path = 'creditcard', train_percentage = 0.8, test_percentage = 0.1)

# Get pointers to Numpy arrays for validation and test data
x_val = dataset.x_val
y_val = dataset.y_val
x_test = dataset.x_test
y_test = dataset.y_test

# If using CCF dataset
# n_inputs = 30
# n_outputs = 2
n_inputs  = dataset._num_features
n_outputs = dataset._num_classes


# Parameters
learning_rate = 0.01
training_epochs = 1000
batch_size = 300
display_step = 100

# Placeholders used to feed the TF model
X = tf.placeholder(tf.float32, [None, n_inputs]) 
y = tf.placeholder(tf.int64, shape=(None)) 

# Set model weights to 0 to start with training
W = tf.Variable(tf.zeros([n_inputs, n_outputs]))
b = tf.Variable(tf.zeros([n_outputs]))

# Construct logits (ie, weighted sum of variables)
logits = tf.matmul(X, W) + b 

softmaxed_logits = tf.nn.softmax(logits)
# We use cross entropy since we have two outputs
cross_entropy = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
# Gradient Descent with AdadeltaOptimizer
#train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# Initializing the variables
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    sess.run(init)
    best_auc = 0
    best_epoch = 0
    start_time = time.time()
    # Training cycle
    for epoch in range(training_epochs):
        total_batch = int(x_test.shape[0]/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = dataset.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            sess.run(train_step, feed_dict={X: batch_xs, y: batch_ys})

        # Display logs per epoch step (using validation data)
        if (epoch+1) % display_step == 0:
            y_score = sess.run(softmaxed_logits, feed_dict={X: x_val})
            y_score = y_score[:,1]
            auc = roc_auc_score(y_true=y_val, y_score=y_score)
            auc *= 100
            if auc > best_auc:
                best_auc = auc
                best_epoch = epoch
            print("Epoch:", '%04d' % (epoch+1), "Validation AUC=", "{:.3f}".format(auc), "Best validation AUC=", "{:.3f}".format(best_auc))

    print("Optimization Finished!")
    y_score = sess.run(softmaxed_logits, feed_dict={X: x_test})
    y_score = y_score[:,1]
    auc = roc_auc_score(y_true=y_test, y_score=y_score)
    auc *= 100
    print("Best validation AUC = ", "{:.3f}".format(best_auc), "Epoch when best AUC was reached = ", best_epoch, "\n")
    print("AUC test (using last trained model) = ", "{:.3f}".format(auc))
    print_time_training(start_time, time.time())