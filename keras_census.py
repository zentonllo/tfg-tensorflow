# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 09:37:10 2017

@author: Alberto Terceño
"""

# Keras example using dummy variables. See how it trains faster than using ML Engine

import pandas as pd
from sklearn.model_selection import train_test_split

from tensorflow.contrib.keras.python.keras.regularizers import l1,l2
from tensorflow.contrib.keras.python.keras.models import Sequential, load_model
from tensorflow.contrib.keras.python.keras.layers import Dense, Dropout, Activation
from tensorflow.contrib.keras.python.keras.constraints import max_norm
from tensorflow.contrib.keras.python.keras.optimizers import RMSprop, Adam
from tensorflow.contrib.keras.python.keras.layers.normalization import BatchNormalization
from tensorflow.contrib.keras.python.keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint, EarlyStopping
from sklearn.metrics import roc_auc_score
import os

def preprocesa_datos(path):
    df = pd.read_csv(path)
    cols_to_transform = [ ' State-gov', ' Bachelors', ' Never-married', ' Adm-clerical', ' Not-in-family', ' White',  ' Male', ' United-States']
    df_with_dummies = pd.get_dummies( data=df, columns = cols_to_transform )
    cols = list(df_with_dummies.columns.values) 
    cols.pop(cols.index(' <=50K')) 
    df_with_dummies = df_with_dummies[cols+[' <=50K']] 
    df = df_with_dummies
    df[' <=50K'] = df[' <=50K'].apply(lambda x: 1 if x == ' <=50K' else 0)
    return df


df_train = preprocesa_datos('./adult.data.csv')

# Disable info warnings from TF
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Hyperparameters
batch_size = 400
epochs = 500
dropout_rate = 0.5

# Parameters for early stopping (increase them when using auc scores)
DELTA = 1e-6
PATIENCE = 200

dataset_tr = df_train.as_matrix()

x_train, x_val, y_train, y_val = train_test_split(dataset_tr[:,:-1], dataset_tr[:,-1], test_size=0.1, random_state=1)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=1)


early_stopping = EarlyStopping(min_delta = DELTA, patience = PATIENCE )

input_dim = dataset_tr.shape[1] - 1
num_classes = 2

model = Sequential()

# Bastante bien con 2 neuronas
model.add(Dense(2,input_shape=(input_dim,), kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Dropout(dropout_rate))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])


history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_val, y_val),
                    callbacks=[early_stopping])


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1], "\n")


y_pred = model.predict_proba(x_test, verbose = 0)
y_score = y_pred[:,1]
auc = roc_auc_score(y_true=y_test, y_score=y_score)
auc *=100
print("Test AUC:", auc)