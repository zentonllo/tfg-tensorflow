# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 13:06:11 2017

@author: Alberto Terceño
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
from numpy import genfromtxt

class Dataset(object):
    
    def __init__(self, path, train_percentage=0.8):
        # Construct a DataSet.
        csv_data = genfromtxt(path, delimiter=',')
        
        x_data = csv_data[:,:-1]
        y_data = csv_data[:,-1]
        
        
        tr_limit = int(x_data.shape[0]*train_percentage)
        
        x_train = x_data[:tr_limit,:]
        y_train = y_data[:tr_limit]

        
        x_test = x_data[tr_limit:,:]
        y_test = y_data[tr_limit:]
        
        
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        

        assert x_train.shape[0] == y_train.shape[0], (
              'x_train.shape: %s y_train.shape: %s' % (x_train.shape, y_train.shape))
        
        self._num_examples = x_train.shape[0]

        self.x_train = x_train
        self.y_train = y_train
        
        self.x_test = x_test
        self.y_test = y_test
        
        self._epochs_completed = 0
        self._index_in_epoch = 0
        
    
    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
          # Finished epoch
          self._epochs_completed += 1
          # Shuffle the data
          perm = np.arange(self._num_examples)
          np.random.shuffle(perm)
          self.x_train = self.x_train[perm]
          self.y_train = self.y_train[perm]
          # Start next epoch
          start = 0
          self._index_in_epoch = batch_size
          assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self.x_train[start:end], self.y_train[start:end]
