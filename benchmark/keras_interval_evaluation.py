# -*- coding: utf-8 -*-
"""
Callback class for Keras which keeps track of AUC over validation data during training

Code mostly adapted from https://github.com/keunwoochoi/keras_callbacks_example
and https://gist.github.com/smly/d29d079100f8d81b905e
"""

from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback


class IntervalEvaluation(Callback):
    """Class used to perform AUC evaluation between some intervals

    Attributes
    ----------
    interval :
        Defines between how many epochs the evaluation is performed  
    X_val :
    	Validation data (Numpy array) without the label column
    y_val :
        Label column (Numpy array) for X_val
    aucs :
        AUC list which saves values periodically

    """
    def __init__(self, validation_data=(), interval=10):
        """__init__ method for the IntervalEvaluation class

        Sets up attributes needed to perform the interval evaluation 

        Parameters
        ----------
        interval :
        		Defines between how many epochs the evaluation is performed  
        X_val :
        	Validation data (Numpy array) without the label column
        y_val :
        	Label column (Numpy array) for X_val
        aucs :
        	AUC list which saves values periodically

        """
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data
        
    def on_train_begin(self, logs={}):
        """Method executed every time a training (fit method) starts"""
        self.aucs = []
        self.losses = []
        return
    
    def on_train_end(self, logs={}):
        """Method executed when the training (fit method) finishes"""
        return

    def on_epoch_end(self, epoch, logs={}):
        """Method executed every time an epoch starts"""
        if epoch % self.interval == 0:
            self.losses.append(logs.get('loss'))
            y_pred = self.model.predict_proba(self.X_val, verbose=0)
            y_pred = y_pred[:,1]
            auc_score = roc_auc_score(y_true=self.y_val, y_score=y_pred)
            auc_score *= 100
            self.aucs.append(auc_score)


            
    def on_batch_begin(self, batch, logs={}):
        """Method executed before a training batch is fed into the model """
        return

    def on_batch_end(self, batch, logs={}):
        """Method executed after a training batch is fed into the model """
        return
