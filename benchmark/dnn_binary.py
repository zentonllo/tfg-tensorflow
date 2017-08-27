"""
Module used to model Deep Neural Networks which solve binary classification problems (1 output neuron)

Code obtained and adapted from:

https://www.tensorflow.org/get_started/
https://github.com/ageron/handson-ml/blob/master/11_deep_learning.ipynb
https://github.com/aymericdamien/TensorFlow-Examples
https://github.com/zentonllo/gcom

"""


import tensorflow as tf
import time
import numpy as np

import os
import matplotlib.pyplot as plt
import itertools

from tensorflow.contrib.layers import fully_connected, dropout
from tensorflow.contrib.framework import arg_scope


from sklearn.metrics import auc, roc_auc_score, roc_curve, confusion_matrix

# Disable info warnings from TF
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


# Source: https://www.tensorflow.org/get_started/summaries_and_tensorboard
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
    """Helper function to print execution times properly formatted."""
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Execution time:","{:0>2}:{:0>2}:{:0>2}".format(int(hours),int(minutes),int(seconds)))

"""
n_inputs: Número de variables de entrada
n_outputs: Número de clases objetivo (2 problemas clasificación binaria, >2 problemas clasificación multiclase)

learning_rate: Tasa de aprendizaje (suele ser 0.001)
hidden_list: Lista de capas ocultas (incluidas neuronas input y outpt, ej: [n_input, 300, 300, n_output])
activation_function: Funciones de activación (ej: tf.nn.relu, tf.nn.elu, tf.nn.sigmoid, tf.nn.tanh, tf.nn.identity)
keep_prob: Keep probability para el Dropout (suele ser 0.5)
regularizer: Regularizer a usar (ej: tf.contrib.layers.l1_regularizer(scale=beta, scope=None), tf.contrib.layers.l2_regularizer(scale=beta, scope=None))

normalizer_fn: función de normalización (None o batch_norm para realizar batch normalization)

normalizer_params = {
    'is_training': None,
    # 0.9 o 0.99 o 0.999 o 0.9999 ...
    # Segun performance guide de TF: menor si va bien en training y peor en validation/test
    # Según A.Geron, aumentar cuando el dataset es grande y los batches pequeños 
    'decay': 0.9,
    'updates_collections': None,
    # Si usamos funciones de activacion que no sean relus --> scale_term debe ser True
    'scale': scale_term,
    # Aumenta rendimiento según la performance guide de TF
    'fused': True
    
    # Try zero_debias_moving_mean=True for improved stability
    # 'zero_debias_moving_mean':True


}

optimizer: = tf.train.AdamOptimizer, tf.train.RMSPropOptimizer, tf.train.AdadeltaOptimizer, tf.train.AdagradOptimizer, tf.train.MomentumOptimizer (este requiere cambios)

El optimizer debe estar instanciado, ej: tf.train.AdamOptimizer(learning_rate=0.001, name='optimizer')

"""



class DNN(object):
    """Class that models a Deep Neural Network with just one output neuron
    
    There are training and predicting methods, as well as tools that generate plots.
    Most of the neural network hyperparameters are set when a class object is instanciated.
    Mostly similar to DNN class in dnn_multiclass (it might be a better way to merge both classes into one)

    
    Attributes
    ----------
    file_writer :
        tf.summary.FileWriter object which adds summaries to TensorBoard 
    saver :
        tf.train.Saver() used to save the model
    merged :
        TF node that if it is executed will generate the TensorBoard summaries
    hidden_list : 
        List with the following shape [input_neurons, neurons_hidden_layer_1, neurons_hidden_layer_2, ..., 1]
    activation_function : 
        TF activation function (tf.nn.relu, tf.nn.elu, tf.nn.sigmoid, tf.nn.tanh, tf.nn.identity, etc.)
    keep_prob :
        Probability to keep a neuron active during dropout (that is, 1 - dropout_rate, use None to avoid dropout)
    regularizer :
        TF regularizer to use (tf.contrib.layers.l1_regularizer(scale=beta, scope=None), tf.contrib.layers.l2_regularizer(scale=beta, scope=None))
    normalizer_fn : 
        Normalizer function to use. Use batch_norm for batch normalization and None to avoid normalizer functions
    normalizer_params : 
        Extra parameters for the normalizer function
    optimizer : 
        TF Optimizer during Gradient Descent (tf.train.AdamOptimizer, tf.train.RMSPropOptimizer, tf.train.AdadeltaOptimizer or tf.train.AdagradOptimizer)
    log_dir : 
        Path used to save all the needed TensorFlow and TensorBoard information to save (graph, models, etc.)
    batch_size : 
        Batch size to be used during training
    y_casted : 
        Label column (NP array) casted to float (casted must be made in order to use the TF cross entropy function)
    predictions : 
        NP array with class predictions (0.5 threshold used)
    """
    
    def __init__(self,
                 log_dir,
                 hidden_list,
                 activation_function=tf.nn.relu,
                 keep_prob = None, 
                 regularizer = None,
                 normalizer_fn = None,
                 normalizer_params = None,
                 optimizer = tf.train.AdamOptimizer(learning_rate=0.001, name='optimizer')
               ):
        
        """__init__ method for the DNN class

        Saves the hyperparameters as attributes and instatiates a deep neural network

        Parameters
        ----------
        log_dir : 
            Path used to save all the needed TensorFlow and TensorBoard information to save (graph, models, etc.)
        hidden_list : 
            List with the following shape [input_neurons, neurons_hidden_layer_1, neurons_hidden_layer_2, ..., 1]
        activation_function : 
            TF activation function (tf.nn.relu, tf.nn.elu, tf.nn.sigmoid, tf.nn.tanh, tf.nn.identity, etc.)
        keep_prob :
            Probability to keep a neuron active during dropout (that is, 1 - dropout_rate, use None to avoid dropout)
        regularizer :
            TF regularizer to use (tf.contrib.layers.l1_regularizer(scale=beta, scope=None), tf.contrib.layers.l2_regularizer(scale=beta, scope=None))
        normalizer_fn : 
            Normalizer function to use. Use batch_norm for batch normalization and None to avoid normalizer functions
        normalizer_params : 
            Extra parameters for the normalizer function
        optimizer : 
            TF Optimizer during Gradient Descent (tf.train.AdamOptimizer, tf.train.RMSPropOptimizer, tf.train.AdadeltaOptimizer or tf.train.AdagradOptimizer)

        """

        # Create a new TF graph from scratch
        tf.reset_default_graph()
        self.file_writer = None
        self.saver = None
        self.merged = None
        self.hidden_list = hidden_list
        self.activation_function = activation_function
        self.keep_prob = keep_prob
        self.regularizer = regularizer
        self.normalizer_fn = normalizer_fn
        self.normalizer_params = normalizer_params
        self.optimizer = optimizer
        self.log_dir = log_dir
        self.batch_size = None
        self.y_casted = None
        self.predictions = None
        
        # Instantiate the neural network
        self.create_net()
    
    def create_net(self):
        """Method that instatiates a neural network using the hyperparameters passed to the DNN object. 
        
        Most of the code was obtained and adapted from 
        https://github.com/ageron/handson-ml/blob/master/11_deep_learning.ipynb
        """

        hidden_list = self.hidden_list
        n_inputs = hidden_list[0]
        # This is hardcoded to show how this class works
        # hidden_list[-1] should be 1 and we should check it out right here
        n_outputs = 1
        
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
                
                # Build the fully-connected layers                
                Z = self.X
                n_iter = len(hidden_list[1:])

                for i in range(1,n_iter):
                    name_scope = "hidden" + str(i)
                    Z = fully_connected(inputs=Z, num_outputs=hidden_list[i], scope=name_scope)
                    if self.keep_prob is not None:
                        Z = dropout(Z, self.keep_prob, is_training=self.is_training)
            
            self.logits = fully_connected(inputs=Z, num_outputs=n_outputs, activation_fn=None, weights_initializer=he_init, normalizer_fn=self.normalizer_fn, normalizer_params=self.normalizer_params, scope="outputs")
            with tf.name_scope("softmaxed_output"):
                self.softmaxed_logits = tf.nn.sigmoid(self.logits)
            
        with tf.name_scope("loss"):
            y_casted = tf.cast(self.y, tf.float32)
            self.y_casted = tf.reshape(y_casted, [-1,1])
            # Compute cross_entropy from logits (that is, dnn output without applying the sigmoid function)
            xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_casted, logits=self.logits)  
            self.loss = tf.reduce_mean(xentropy)
            if self.regularizer is not None:
                self.loss += tf.contrib.layers.apply_regularization(self.regularizer, tf.trainable_variables())
        tf.summary.scalar('cross_entropy', self.loss)               
                
            
        with tf.name_scope("train"):
            opt = self.optimizer
            # Minimize the loss function
            self.train_step = opt.minimize(self.loss, name='train_step')

        with tf.name_scope("eval"):
            self.predictions = tf.round(self.softmaxed_logits)
            incorrect = tf.abs(tf.subtract(self.predictions, self.y_casted))
            incorrect_casted = tf.cast(incorrect, tf.float32)
            self.accuracy = tf.subtract(tf.cast(100, tf.float32),tf.reduce_mean(incorrect_casted))
        tf.summary.scalar('accuracy', self.accuracy)
        
        # TensorBoard summaries for the hidden layers weights
        for i in range(1,n_iter):
            with tf.variable_scope('hidden'+str(i), reuse=True):
                variable_summaries(tf.get_variable('weights'))
                
        with tf.variable_scope('outputs', reuse=True):
                variable_summaries(tf.get_variable('weights'))


        self.merged = tf.summary.merge_all()
        self.init = tf.global_variables_initializer()
    
        self.saver = tf.train.Saver()
    
    
    def feed_dict(self, dataset, mode):
        """Method that builds a dictionary to feed the neuronal network. 

        Parameters
        ----------
        dataset :
            Dataset object
        mode : 
            String that points which feed dictionary we want to get. Possible values: 'batch_training', 'training_test', 'validation_test'

        Returns
        -------
        fd
            Dictionary that feeds the TensorFlow model
        """

        fd = None
        if mode is 'batch_training':
            x_batch, y_batch = dataset.next_batch(self.batch_size)
            fd = {self.is_training: True, self.X: x_batch, self.y: y_batch}
        elif mode is 'training_test':
            fd = {self.is_training: False, self.X: dataset.x_train, self.y: dataset.y_train}
        elif mode is 'validation_test':
            fd = {self.is_training: False, self.X: dataset.x_val, self.y: dataset.y_val}

        return fd
    
    def train(self, dataset, model_path, train_path, nb_epochs=100, batch_size=10, silent_mode=False):
        """Method that trains a deep neuronal network. 

        Parameters
        ----------
        dataset :
            Dataset object
        model_path : 
            Path where the optimal TensorFlow model will be saved
        train_path : 
            Path where the training TensorFlow models will be saved. After the training process, the model trained in the very last epoch will
            be the only one saved
        nb_epochs : 
            Number of epochs to train the model
        batch_size : 
            Batch size to be used during training
        silent_mode : 
            Flag which enables whether to print progress on the terminal during training.

        Returns
        -------
        None
        """

        start_time = time.time()
        
        x_training = dataset.x_train
        y_training = dataset.y_train
        
        x_validation = dataset.x_val
        y_validation = dataset.y_val
        
        nb_data = dataset._num_examples
        # nb_batches = nb_data // batch_size (integer division)
        self.batch_size= batch_size
        nb_batches = int(nb_data/batch_size)
        
        # Records best validation AUC during training, which will allow to save that model as optimal
        best_auc = 0
        self.aucs = []

        with tf.Session() as sess:
            sess.run(self.init)
            
            self.file_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
            for epoch in range(nb_epochs):
                # Iterate through batches and keep training
                for batch in range(nb_batches):
                    sess.run(self.train_step,
                             feed_dict=self.feed_dict(dataset, mode='batch_training'))
                self.saver.save(sess, train_path)
                
                # Get the summaries for TensorBoard
                summary = sess.run(self.merged, feed_dict=self.feed_dict(dataset, mode='training_test'))
                self.file_writer.add_summary(summary, epoch)
                
                # We use a sklearn function to compute AUC. Couldn't manage to make tf.metrics.auc work due to some odd 'local variables'
                cur_auc = self.auc_roc(x_validation, y_validation, train_path)
                summary_auc = tf.Summary(value=[tf.Summary.Value(tag="AUCs_Validation", simple_value=cur_auc)])
                self.file_writer.add_summary(summary_auc, epoch)
                
                # Only save best model if it gets the best AUC over the validation set
                if cur_auc > best_auc:
                    best_auc = cur_auc
                    self.saver.save(sess, model_path)
                
                if not silent_mode:
                    acc_train = sess.run(self.accuracy, feed_dict=self.feed_dict(dataset, mode='training_test'))
                    auc_train = self.auc_roc(x_training, y_training, train_path)
                    acc_val = sess.run(self.accuracy, feed_dict=self.feed_dict(dataset, mode='validation_test'))
                    print("Epoch:", (epoch+1), "Train accuracy:", acc_train, "Train AUC:", auc_train )
                    print("Validation accuracy:", acc_val, "Validation AUC:", cur_auc, "Best Validation AUC:", best_auc, "\n")

        
        self.file_writer.close()
        print_execution_time(start_time, time.time())
        
        if silent_mode:
            print("Best Validation AUC:", best_auc)
            
    
    def predict(self, x_test, model_path):
        """Method that gets predictions from a trained deep neuronal network. 

        Get a Numpy array of predictions P(y=1| W) for all the x_test

        Parameters
        ----------
        x_test :
            Numpy array with data test to get predictions for
        model_path : 
            Path where the TensorFlow model is located

        Returns
        -------
        Numpy array with predictions (probabilities between 0 and 1)
        """
        with tf.Session() as sess:
            self.saver.restore(sess, model_path)
            y_pred = sess.run(self.softmaxed_logits, feed_dict={self.is_training: False, self.X: x_test})
        return y_pred
    
    def predict_class(self, x_test, model_path):
        """Method that gets predicted classes (0 or 1) from a trained deep neuronal network. 

        Get a Numpy array of predicted classes for all the x_test

        Parameters
        ----------
        x_test :
            Numpy array with data test to get their predicted classes
        model_path : 
            Path where the TensorFlow model is located

        Returns
        -------
        Numpy array with predicted classes (0 or 1)
        """
        with tf.Session() as sess:
            self.saver.restore(sess, model_path)
            y_pred = sess.run(self.predictions, feed_dict={self.is_training: False, self.X: x_test})
        return y_pred
    
    def test(self, x_test, y_test, model_path):
        """Method that prints accuracy and AUC for test data after getting predictions a trained deep neuronal network. 


        Parameters
        ----------
        x_test :
            Numpy array with data test to get their predicted classes
        y_test : 
            Numpy array with the labels belonging to x_test
        model_path : 
            Path where the TensorFlow model is located

        Returns
        -------
        None
        """
        start_time = time.time()
        with tf.Session() as sess:
            self.saver.restore(sess, model_path)
            acc_test = sess.run(self.accuracy, feed_dict={self.is_training: False, self.X: x_test, self.y: y_test})
            print("Test accuracy:", acc_test)
            auc_test = self.auc_roc(x_test, y_test, model_path)
            print("Test AUC:", auc_test)
            
        print_execution_time(start_time, time.time())


    def auc_roc(self, x_test, y_test, model_path):
        """Method that computes AUC for some data after getting predictions from a trained deep neural network. 


        Parameters
        ----------
        x_test :
            Numpy array with data test to get their predicted classes
        y_test : 
            Numpy array with the labels belonging to x_test
        model_path : 
            Path where the TensorFlow model is located

        Returns
        -------
        AUC value for the test data (x_test and y_test)
        """
        y_score = self.predict(x_test, model_path)
        auc = roc_auc_score(y_true=y_test, y_score=y_score)
        return auc*100
    
    
    def save_roc(self, x_test, y_test, model_path, roc_path):
        """Method that computes a ROC curve from a model and save it as a png file. 


        Parameters
        ----------
        x_test :
            Numpy array with data test to get their predicted classes
        y_test : 
            Numpy array with the labels belonging to x_test
        model_path : 
            Path where the TensorFlow model is located
        roc_path :
            Path that points where to save the png file with the ROC curve

        Returns
        -------
        None
        """
        y_score = self.predict(x_test, model_path)
        fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_score)
        roc_auc = auc(fpr, tpr)
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc*100)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(roc_path, bbox_inches='tight')
        
        
    def save_cm(self, x_test, y_test, model_path, cm_path, classes, normalize=True):
        """Method that computes a confusion matrix from a model and save it as a png file. 


        Parameters
        ----------
        x_test :
            Numpy array with data test to get their predicted classes
        y_test : 
            Numpy array with the labels belonging to x_test
        model_path : 
            Path where the TensorFlow model is located
        cm_path :
            Path that points where to save the png file with the confusion matrix
        classes : 
            List with labels for the confusion matrix rows and columns. For instance: ['Normal Transactions', 'Fraudulent transactions']

        Returns
        -------
        None
        """
        y_pred = self.predict_class(x_test, model_path)
        cm = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=2)
        plt.figure()
        cmap=plt.cm.Blues
        if normalize:
          plt.title('Normalized confusion matrix') 
        else:
          plt.title('Confusion matrix, without normalization')   
            
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        plt.savefig(cm_path, bbox_inches='tight')
        

    