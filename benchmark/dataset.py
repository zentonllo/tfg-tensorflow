# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 13:06:11 2017

@author: Alberto Terceño

Module used for data ingestion in the benchmark. 
It can read csv and npy datasets with numerical features, whose label column is the last one.

Code used to start coding the class was obtained and adapted from TF MNIST data reader: 
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import pandas as pd
from os.path import isfile, abspath

class Dataset(object):
    """Class used to model a dataset in Python

    It allows to read datasets from csv and npy files, request batches and some general
    information about the dataset.
    This class also splits the dataset into training, validation and test sets 


    Attributes
    ----------
    _num_examples :
        Number of rows in the dataset 
    _num_features :
        Number of features in the dataset (not including the label column)
    _num_classes :
        Number of target classes (2 for binary problems and >2 for the multiclass ones)
    _epochs_completed :
        Number of epochs completed, that is, how many times the whole dataset has been returned via next batch requests
    _index_in_epoch :
        Index used to keep track of the current dataset batch
    x_train :
        Numpy array used as the training set (without the label column)
    y_train :
        Label column (Numpy array) for the x_train attribute
    x_val :
        Numpy array used as the validation set (without the label column)
    y_val :
        Label column (Numpy array) for the x_val attribute
    x_test :
        Numpy array used as the test set (without the label column)
    y_test :
        Label column (Numpy array) for the x_test attribute

    """

    def __init__(self, path, train_percentage=0.8, test_percentage=0.1):

        """__init__ method for the Dataset class

        Reads a dataset and performs the split into training, validation and test sets according to the split parameters. 
        It finally sets up the class attributes. 

        Parameters
        ----------
        path :
           Windows/Unix path for the csv or npy dataset. Do not include the dataset extension!
        train_percentage :
           Percentage of rows to be used as training set. Use train_percentage = 0 if using the dataset for testing purposes only.
           train_percentage + test_percentage should be less or equal to one
        test_percentage :
           Percentage of rows to be used as test set. train_percentage + test_percentage should be less or equal to one
        """

        # We use the spare percentage for the validation set
        val_percentage = 1 - (train_percentage + test_percentage)
        assert (train_percentage+test_percentage+val_percentage)==1, (
                'train_percentage: %s, validation_percentage: %s, test_percentage: %s' % (train_percentage*100, val_percentage*100, test_percentage*100) )



        np_file =  abspath(path + '.npy')
        csv_file = abspath(path + '.csv')


        data = None
        # Check if the Numpy file exists. If not, it is generated, so we can get delete the csv file.
        if isfile(np_file):
            data = np.load(np_file)
        else:
        	# We skip the header (first row) since we just need the data itself as a Numpy array
            data = pd.read_csv(csv_file, header=None, skiprows = 1).as_matrix()
            np.save(np_file, data)

        # Shuffling the dataset
        np.random.shuffle(data)

        # Split the dataset
        x_data = data[:,:-1]
        y_data = data[:,-1]

        dataset_samples = x_data.shape[0]

        tr_limit = int(dataset_samples*train_percentage)

        x_train = x_data[:tr_limit,:]
        y_train = y_data[:tr_limit]

        val_limit = int(dataset_samples*(train_percentage+val_percentage))

        x_val = x_data[tr_limit:val_limit,:]
        y_val = y_data[tr_limit:val_limit]

        x_test = x_data[val_limit:,:]
        y_test = y_data[val_limit:]

        # We just treat numerical features. This cast is made in order to use
        # the same type in the benchmark scripts
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')


        assert x_train.shape[0] == y_train.shape[0], (
              'x_train.shape: %s y_train.shape: %s' % (x_train.shape, y_train.shape))

        self._num_examples = x_train.shape[0]
        self._num_features = x_train.shape[1]
        self._num_classes = len(np.unique(y_data))

        self.x_train = x_train
        self.y_train = y_train

        self.x_val = x_val
        self.y_val = y_val

        self.x_test = x_test
        self.y_test = y_test

        self._epochs_completed = 0
        self._index_in_epoch = 0


    def next_batch(self, batch_size):
        """ Method returning a certain batch of data. 
        Most of the code was obtained and adapted from 
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py

        Parameters
        ----------
        batch_size :
            Number of rows returned from the training set

        Returns
        -------
        x_train, y_train
            Training batch to feed the models during training
        """
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
          # Done with the current epoch
          self._epochs_completed += 1
          # Shuffle the data when an epoch is completed
          perm = np.arange(self._num_examples)
          np.random.shuffle(perm)
          self.x_train = self.x_train[perm]
          self.y_train = self.y_train[perm]
          # Reset index to start a new epoch
          start = 0
          self._index_in_epoch = batch_size
          assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self.x_train[start:end], self.y_train[start:end]
