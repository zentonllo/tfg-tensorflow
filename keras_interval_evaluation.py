# -*- coding: utf-8 -*-
"""
Idea got from https://github.com/keunwoochoi/keras_callbacks_example
and https://gist.github.com/smly/d29d079100f8d81b905e
"""

from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback


class IntervalEvaluation(Callback):
    def __init__(self, validation_data=(), interval=10):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data
        
    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []
        return
    
    def on_train_end(self, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            self.losses.append(logs.get('loss'))
            y_pred = self.model.predict_proba(self.X_val, verbose=0)
            y_pred = y_pred[:,1]
            auc_score = roc_auc_score(y_true=self.y_val, y_score=y_pred)
            auc_score *= 100
            self.aucs.append(auc_score)


            
    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
