import tensorflow as tf
import time
import numpy as np
from dataset import Dataset

# Desactivamos warnings
import os


import sys
from tensorflow.contrib.layers import fully_connected, batch_norm, dropout
from tensorflow.contrib.framework import arg_scope
from leaky_relu import leaky_relu

from datetime import datetime
from sklearn.metrics import roc_auc_score

# Disable info warnings from TF
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


NOW = datetime.utcnow().strftime("%Y%m%d%H%M%S")
#ROOT_LOGDIR = 'tf_logs'
ROOT_LOGDIR = 'C:/tmp'
LOG_DIR = "{}/run-{}".format(ROOT_LOGDIR, NOW)

# Checkpoints default paths
M_PATH = "./model/DNN.ckpt"
TR_PATH = "./training/DNN_tr.ckpt"




"""
n_inputs = 28 * 28  # Número de variables de entrada
n_outputs = 10

learning_rate = 0.001
hidden_list = [n_input, 300, 300, n_output]
activation_function = tf.nn.relu, tf.nn.elu, tf.nn.sigmoid, tf.nn.tanh, tf.nn.identity
keep_prob = 0.5
regularizer = tf.contrib.layers.l1_regularizer, tf.contrib.layers.l2_regularizer

normalizer_fn=batch_norm
# "batch_norm_params"
normalizer_params = {
    'is_training': is_training,
    'decay': 0.9,
    'updates_collections': None,
    'scale': True,
}

optimizer = tf.train.AdamOptimizer, tf.train.RMSPropOptimizer, tf.train.AdadeltaOptimizer, tf.train.AdagradOptimizer, tf.train.MomentumOptimizer (este requiere cambios)
"""

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def print_execution_time(start, end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Execution time:","{:0>2}:{:0>2}:{:0>2}".format(int(hours),int(minutes),int(seconds)))
    



# 2 outputs para problemas de clasificación binaria
class DNN(object):
    
    
    def __init__(self,
                 hidden_list,
                 activation_function=tf.nn.relu,
                 keep_prob = None, 
                 regularizer = None, 
                 beta = 0,
                 normalizer_fn = None,
                 normalizer_params = None,
                 optimizer = tf.train.AdamOptimizer(learning_rate=0.001, name='optimizer')
               ):
        
        tf.reset_default_graph()
        self.file_writer = None
        self.saver = None
        self.learning_rate = 0.001
        self.hidden_list = hidden_list
        self.activation_function = activation_function
        self.keep_prob = keep_prob
        self.regularizer = None
        if regularizer is not None:
            self.regularizer = regularizer(scale=beta,scope=None)
        self.normalizer_fn = normalizer_fn
        self.normalizer_params = normalizer_params
        self.optimizer = optimizer
        
        self.create_net()
    
    def create_net(self):
        hidden_list = self.hidden_list
        n_inputs = hidden_list[0]
        n_outputs = hidden_list[-1]
        
        self.X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
        self.y = tf.placeholder(tf.int64, shape=(None), name="y")
        self.is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
        
        if self.normalizer_params is not None:
            self.normalizer_params['is_training'] = self.is_training
            
        with tf.name_scope("dnn"):
            he_init = tf.contrib.layers.variance_scaling_initializer()
        
            with arg_scope(
                    [fully_connected],
                    activation_fn=self.activation_function,
                    weights_initializer=he_init,
                    normalizer_fn=self.normalizer_fn,
                    normalizer_params=self.normalizer_params):
                
                Z = self.X
                n_iter = len(hidden_list[1:])
                for i in range(1,n_iter):
                    name_scope = "hidden" + str(i)
                    Z = fully_connected(Z, hidden_list[i], scope=name_scope)
                    if self.keep_prob is not None:
                        Z = dropout(Z, self.keep_prob, is_training=self.is_training)
                    
            self.logits = fully_connected(Z, n_outputs, activation_fn=None, scope="outputs")
            self.softmaxed_logits = tf.nn.softmax(self.logits)     
            
        with tf.name_scope("loss"):
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits) 
            self.loss = tf.reduce_mean(xentropy)
            if self.regularizer is not None:
                self.loss += tf.contrib.layers.apply_regularization(self.regularizer, tf.trainable_variables())
        tf.summary.scalar('cross_entropy', self.loss)               
                
            
        with tf.name_scope("train"):
            opt = self.optimizer
            self.train_step = opt.minimize(self.loss, name='train_step')

        with tf.name_scope("eval"):
            # Ver si esto es correcto, creo que habría que poner  self.softmaxed_logits
            # Antigua versión
            # correct = tf.nn.in_top_k(self.logits,self.y, 1)
            # Nueva versión
            correct = tf.nn.in_top_k(self.softmaxed_logits,self.y, 1)
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)
        
        self.merged = tf.summary.merge_all()
        self.init = tf.global_variables_initializer()
    
        self.saver = tf.train.Saver()
        
    def train(self, dataset, nb_epochs=100, batch_size=10, model_path=M_PATH, train_path=TR_PATH, silent_mode=False):
        
        start_time = time.time()
        
        x_training = dataset.x_train
        y_training = dataset.y_train
        
        nb_data = x_training.shape[0]
        # nb_batches = nb_data // batch_size
        nb_batches = int(nb_data/batch_size)
        
        best_auc = 0

        with tf.Session() as sess:
            sess.run(self.init)
            
            self.file_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
            for epoch in range(nb_epochs):
                for batch in range(nb_batches):
                    x_batch, y_batch = dataset.next_batch(batch_size)
                    sess.run(self.train_step,
                             feed_dict={self.is_training: True, self.X: x_batch,
                                        self.y: y_batch})
                self.saver.save(sess, train_path)
                #Check de esto
                summary = sess.run(self.merged, feed_dict={self.is_training: False, self.X: x_training, self.y: y_training})
                self.file_writer.add_summary(summary, epoch)
                cur_auc = self.auc_roc(dataset.x_test, dataset.y_test, train_path)
                if cur_auc > best_auc:
                    best_auc = cur_auc
                    self.saver.save(sess, model_path)
                acc_train = sess.run(self.accuracy, feed_dict={self.is_training: False, self.X: x_training, self.y: y_training})
                
                if not silent_mode:
                    print("Epoch:", (epoch+1), "Train accuracy:", acc_train, "AUC:", cur_auc, "Best AUC:", best_auc)
                    print("Train AUC:", self.auc_roc(dataset.x_train, dataset.y_train, train_path))
        
        self.file_writer.close()
        print_execution_time(start_time, time.time())
        if silent_mode:
            print("AUC:", best_auc)
        
        
    def predict(self, x_test, model_path=M_PATH):
        with tf.Session() as sess:
            self.saver.restore(sess, model_path)
            y_pred = sess.run(self.softmaxed_logits, feed_dict={self.is_training: False, self.X: x_test})
        return y_pred[:,1]

    def test(self, x_test, y_test, model_path=M_PATH):
        with tf.Session() as sess:
            self.saver.restore(sess, model_path)
            acc_test = sess.run(self.accuracy, feed_dict={self.is_training: False, self.X: x_test, self.y: y_test})
            print("Test accuracy:", acc_test)


    def auc_roc(self, x_test, y_test, model_path=M_PATH):
        y_score = self.predict(x_test, model_path)
        auc = roc_auc_score(y_true=y_test, y_score=y_score)
        return auc*100
    
    

if __name__ == "__main__":
    
    """    
    OUTPUT_FILE = "results_dnn.txt"
    sys.stdout = open(OUTPUT_FILE, "w")
    """
    
    # Carga del dataset 
    # En R hacemos previamente: write.table(MyData, file = "MyData.csv",row.names=FALSE, na="",col.names=FALSE, sep=",")
    
    #OJO Muy gran training set!
    dataset = Dataset(path = 'validation_it17.csv', train_percentage = 0.8 )
    #dataset = Dataset(path = 'data_regf_it17.csv', train_percentage = 0.8 )
    x_test = dataset.x_test
    y_test = dataset.y_test
    
    
    #n_inputs = 65
    #n_outputs = 2
    n_inputs  = x_test.shape[1]
    n_outputs = np.unique(y_test).size
    
    # Hyperparameters 
    learning_rate = 0.001
    hidden_list = [n_inputs, 5, n_outputs]
    activation_function = tf.nn.elu
    #activation_function = leaky_relu
    #keep_prob = 0.7
    keep_prob = None
    
    
    nb_epochs = 200
    #batch_size = 500
    batch_size = 10000
    
    
    #regularizer = tf.contrib.layers.l2_regularizer
    #beta = 0.001
    regularizer = None
    beta = None
    
    """
    normalizer_fn = None
    normalizer_params = None
    """
    normalizer_fn=batch_norm
    # "batch_norm_params"
    normalizer_params = {
        'is_training': None,
        'decay': 0.9,
        'updates_collections': None,
        'scale': True,
    }
    
    
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, name='optimizer')
    #optimizer = tf.train.AdagradOptimizer(learning_rate=0.001, name='optimizer')
    #optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, name='optimizer')

    dnn = DNN(hidden_list=hidden_list,
              activation_function = activation_function,
              keep_prob = keep_prob,
              regularizer = regularizer,
              beta = beta,
              normalizer_fn = normalizer_fn,
              normalizer_params = normalizer_params,
              optimizer = optimizer)
    

    
    dnn.train(dataset, nb_epochs, batch_size)
    dnn.test(x_test=x_test, y_test=y_test)

    """
    print("------------Validation------------","\n")
    start_validation = time.time()
    dataset_val = Dataset(path = 'validation_it17.csv', train_percentage = 0 )
    dnn.test(x_test=dataset_val.x_test, y_test=dataset_val.y_test)
    auc_val = dnn.auc_roc(x_test=dataset_val.x_test, y_test=dataset_val.y_test)
    print("Validation AUC:", auc_val)
    print_execution_time(start_validation, time.time())
    """
    
    
    """
    sys.stdout = sys.__stdout__
    """