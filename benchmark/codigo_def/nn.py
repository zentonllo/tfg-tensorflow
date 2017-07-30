"""
Obtenido a partir de ejemplos de tensorflow.org y del libro de Aurelien Geron sobre Machine Learning


Author: Alberto Terceño Ortega
"""

"""
TODO


-- Fork de los dnn binarios y multiclase
-- Pruebas con un dnn binario con una neurona al final y con función sigmoide (rendimiento?)


--------------------------------
-- Investigar ejemplos ML engine (census)
-- Investigar api python google analytics


--------------------------------
-- Trabajo futuro

-- Incluir tsne en el playground (scikit learn y luego pasar al embedding de Tensorboard)
-- Codificación de variables cualitativas a cuantitativas
-- Investigar Python VTreat
-- Añadir interval evaluation en Keras y funcionalidades extra (BN, etc...)
-- Cross Validation 
-- División elegante en módulos como ejemplos de MNIST
-- Usar pandas en un jupyter notebook para preprocesar csv
-- Incluir local response normalization y data augmentation?
-- Imágenes png pasarlas a tensorboard (gyglim)
-- Se podría usar tf.metrics.auc sin inicializar variables locales?



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


# 2 outputs para problemas de clasificación binaria
class DNN(object):
    
    
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
        
        tf.reset_default_graph()
        self.file_writer = None
        self.saver = None
        self.merged = None
        self.learning_rate = 0.001
        self.hidden_list = hidden_list
        self.activation_function = activation_function
        self.keep_prob = keep_prob
        self.regularizer = regularizer
        self.normalizer_fn = normalizer_fn
        self.normalizer_params = normalizer_params
        self.optimizer = optimizer
        self.log_dir = log_dir
        self.batch_size = None
        
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
                    Z = fully_connected(inputs=Z, num_outputs=hidden_list[i], scope=name_scope)
                    if self.keep_prob is not None:
                        Z = dropout(Z, self.keep_prob, is_training=self.is_training)
            
            # Batch normalization en la ultima capa?
            self.logits = fully_connected(inputs=Z, num_outputs=n_outputs, activation_fn=None, weights_initializer=he_init, normalizer_fn=self.normalizer_fn, normalizer_params=self.normalizer_params, scope="outputs")
            #self.logits = fully_connected(inputs=Z, num_outputs=n_outputs, activation_fn=None, weights_initializer=he_init, scope="outputs")
            with tf.name_scope("softmaxed_output"):
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
            correct = tf.nn.in_top_k(self.softmaxed_logits,self.y, 1)
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)
        
        # Summaries en Tensorboard de los pesos de las capas de la red neuronal
        for i in range(1,n_iter):
            with tf.variable_scope('hidden'+str(i), reuse=True):
                variable_summaries(tf.get_variable('weights'))
                
        with tf.variable_scope('outputs', reuse=True):
                variable_summaries(tf.get_variable('weights'))


        self.merged = tf.summary.merge_all()
        self.init = tf.global_variables_initializer()
    
        self.saver = tf.train.Saver()
    
    # Wrapper para el training
    def feed_dict(self, dataset, mode):
        
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
        
        start_time = time.time()
        
        x_training = dataset.x_train
        y_training = dataset.y_train
        
        x_validation = dataset.x_val
        y_validation = dataset.y_val
        
        nb_data = dataset._num_examples
        # nb_batches = nb_data // batch_size
        self.batch_size= batch_size
        nb_batches = int(nb_data/batch_size)
        
        best_auc = 0
        self.aucs = []

        with tf.Session() as sess:
            sess.run(self.init)
            
            self.file_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
            for epoch in range(nb_epochs):
                for batch in range(nb_batches):
                    sess.run(self.train_step,
                             feed_dict=self.feed_dict(dataset, mode='batch_training'))
                self.saver.save(sess, train_path)
                
                # Correr los summaries con los datos del batch?
                summary = sess.run(self.merged, feed_dict=self.feed_dict(dataset, mode='training_test'))
                self.file_writer.add_summary(summary, epoch)
                
                # Añadimos el auc_roc sobre validacion de esta manera, pues usar tf.metrics.auc requiere inicializar variables locales, lo cual no he logrado conseguir en el código
                cur_auc = self.auc_roc(x_validation, y_validation, train_path)
                summary_auc = tf.Summary(value=[tf.Summary.Value(tag="AUCs_Validation", simple_value=cur_auc)])
                self.file_writer.add_summary(summary_auc, epoch)
                

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
            
    
    # Da la probabilidad de num_class para los x_test
    def predict(self, x_test, model_path, num_class=1):
        with tf.Session() as sess:
            self.saver.restore(sess, model_path)
            y_pred = sess.run(self.softmaxed_logits, feed_dict={self.is_training: False, self.X: x_test})
        return y_pred[:,num_class]
    
    # Predice la clase para los x_test (es decir, toma el máximo de las probabilidades de salida)
    def predict_class(self, x_test, model_path):
        with tf.Session() as sess:
            self.saver.restore(sess, model_path)
            y_pred = sess.run(self.softmaxed_logits, feed_dict={self.is_training: False, self.X: x_test})
            return tf.argmax(y_pred,1).eval()
    
    def test(self, x_test, y_test, model_path):
        start_time = time.time()
        with tf.Session() as sess:
            self.saver.restore(sess, model_path)
            acc_test = sess.run(self.accuracy, feed_dict={self.is_training: False, self.X: x_test, self.y: y_test})
            print("Test accuracy:", acc_test)
            auc_test = self.auc_roc(x_test, y_test, model_path)
            print("Test AUC:", auc_test)
            
        print_execution_time(start_time, time.time())


    def auc_roc(self, x_test, y_test, model_path):
        y_score = self.predict(x_test, model_path)
        auc = roc_auc_score(y_true=y_test, y_score=y_score)
        return auc*100
    
    
    def save_roc(self, x_test, y_test, model_path, roc_path):
        y_score = self.predict(x_test, model_path)
        fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_score)
        roc_auc = auc(fpr, tpr)
        roc_auc *= 100
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(roc_path, bbox_inches='tight')
        
        

    # Classes: vector con strings para las clases
    def save_cm(self, x_test, y_test, model_path, cm_path, classes, normalize=True):
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
        

if __name__ == "__main__":
    # Test using a Kaggle dataset (credit card fraud detection) 
    # with hardcoded hyperparams in a playground
    # URL: https://www.kaggle.com/dalpozz/creditcardfraud
    exec(open('test_playground.py').read())
    
    
    
"""

def _get_streaming_metrics(prediction,label,num_classes):

    with tf.name_scope("test"):
        # the streaming accuracy (lookup and update tensors)
        accuracy,accuracy_update = tf.metrics.accuracy(label, prediction, 
                                               name='accuracy')
        # Compute a per-batch confusion
        batch_confusion = tf.confusion_matrix(label, prediction,
                                             num_classes=num_classes,
                                             name='batch_confusion')
        # Create an accumulator variable to hold the counts
        confusion = tf.Variable( tf.zeros([num_classes,num_classes], 
                                          dtype=tf.int32 ),
                                 name='confusion' )
        # Create the update op for doing a "+=" accumulation on the batch
        confusion_update = confusion.assign( confusion + batch_confusion )
        # Cast counts to float so tf.summary.image renormalizes to [0,255]
        confusion_image = tf.reshape( tf.cast( confusion, tf.float32),
                                  [1, num_classes, num_classes, 1])
        # Combine streaming accuracy and confusion matrix updates in one op
        test_op = tf.group(accuracy_update, confusion_update)

        tf.summary.image('confusion',confusion_image)
        tf.summary.scalar('accuracy',accuracy)

    return test_op,accuracy,confusion


"""