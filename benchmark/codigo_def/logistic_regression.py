'''
A logistic regression learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import tensorflow as tf
import numpy as np
import time
from dataset import Dataset

# Desactivamos warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from sklearn.metrics import roc_auc_score


def print_time_training(start, end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Training time:","{:0>2}:{:0>2}:{:0>2}".format(int(hours),int(minutes),int(seconds)))


# Carga del dataset 
# En R hacemos previamente: write.table(MyData, file = "MyData.csv",row.names=FALSE, na="",col.names=FALSE, sep=",")
dataset = Dataset(path = 'creditcards', train_percentage = 0.8 )
x_test = dataset.x_test
y_test = dataset.y_test


#n_inputs = 65
#n_outputs = 2
n_inputs  = x_test.shape[1]
n_outputs = np.unique(y_test).size


# Parameters
learning_rate = 0.1
training_epochs = 2000
batch_size = 300
display_step = 100

# tf Graph Input
X = tf.placeholder(tf.float32, [None, n_inputs]) 
y = tf.placeholder(tf.int64, shape=(None)) 

# Set model weights
W = tf.Variable(tf.zeros([n_inputs, n_outputs]))
b = tf.Variable(tf.zeros([n_outputs]))

# Construct logits
logits = tf.matmul(X, W) + b 

softmaxed_logits = tf.nn.softmax(logits)
# Minimize error using cross entropy
cross_entropy = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
# Gradient Descent
train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(cross_entropy)
#train_step = tf.train.MomentumOptimizer(learning_rate,0.5).minimize(cross_entropy)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    best_auc = 0
    start_time = time.time()
    # Training cycle
    for epoch in range(training_epochs):
        total_batch = int(x_test.shape[0]/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = dataset.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            sess.run(train_step, feed_dict={X: batch_xs, y: batch_ys})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            y_score = sess.run(softmaxed_logits, feed_dict={X: x_test})
            y_score = y_score[:,1]
            auc = roc_auc_score(y_true=y_test, y_score=y_score)
            auc *= 100
            if auc > best_auc:
                best_auc = auc
            print("Epoch:", '%04d' % (epoch+1), "AUC=", "{:.9f}".format(auc), "Best AUC=", "{:.9f}".format(best_auc))

    print("Optimization Finished!")
    print_time_training(start_time, time.time())