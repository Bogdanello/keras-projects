import matplotlib.pyplot as plt
import pandas
import numpy as np
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import TimeDistributedDense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical

# Xtrain = matrix representing time points
# Ytrain = matrix representing labels
# embedding_size = dimensionality of embedding output (default 128)
# max_features = 1 + the maximum value amid the features
# dense_size = the size of the dense model (default 5)
# nb_epoch = the number of training epochs (default 2)
def train_simple_lstm(Xtrain, Ytrain, **kwargs):
    max_features = kwargs.get("max_features", Xtrain.max() + 1)
    embedding_size = kwargs.get("embedding_size", 128)
    dense_size = kwargs.get("dense_size", 5)
    n_epoch = kwargs.get("nb_epoch", 2)
    Ytrain = to_categorical(Ytrain.reshape(Ytrain.shape[0],1))
    
    model = Sequential()
    model.add(Embedding(max_features, embedding_size, input_length=Xtrain.shape[1], dropout=0.2))
    model.add(LSTM(embedding_size))
    model.add(Dense(dense_size))
    model.add(Activation('sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(Xtrain, Ytrain, nb_epoch=n_epoch)
    return model

dataset = np.loadtxt("ElectricDevices_TRAIN", delimiter=",")
trainX = dataset[:,1:95]
trainY = dataset[:,0].astype(int)

train_simple_lstm(trainX, trainY)
# model = Sequential()
# model.add(LSTM(4, input_dim=96))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(trainX, trainY, nb_epoch=100, batch_size=96, verbose=2)

