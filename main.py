
from util import split_sequence
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import schedules
import numpy as np
from numpy import array
import random
from random import randint
import os
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Activation, MaxPooling1D, Dropout
from train import trainModel
from test import testing
#from keras.optimizers import SGD

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1" #model will be trained on GPU 1

"""Hyperparameters"""
#dataPath = "data/sinewave.csv"
#dataPath = "data/kdd/corrected"
#n_features = 42          # Univariate time series
#dataPath = "data/sinewave.csv"
# dataPath = "data/sinewave.csv"
# n_features = 2 # Univariate time series
dataPath = "data/AirQualityUCI/preprocessAir.csv"
n_features = 13 # Univariate time series
w = 7
print("dataPath ",dataPath)

                         # i.e., filter(kernel) size       
p_w = 1# Prediction window (number of time stampes required to be 
                         # predicted)

kernel_size = 2          # Size of filter in conv layers
num_filt_1 = 32          # Number of filters in first conv layer
num_filt_2 = 32          # Number of filters in second conv layer
num_nrn_dl = 40          # Number of neurons in dense layer
num_nrn_ol = p_w         # Number of neurons in output layer

conv_strides = 1
pool_size_1 = 2          # Length of window of pooling layer 1
pool_size_2 = 2          # Length of window of pooling layer 2
pool_strides_1 = 2       # Stride of window of pooling layer 1
pool_strides_2 = 2       # Stride of window of pooling layer 2

epochs = 1000
dropout_rate = 0.5       # Dropout rate in the fully connected layer
learning_rate = 2e-5  
anm_det_thr = 0.8        # Threshold for classifying anomaly (0.5~0.8)

import sys
if len(sys.argv) < 2:
    print("usage: python main.py train|test")
else:
    if sys.argv[1] == "test":
        testingObj = testing(dataPath,w ,p_w,n_features,kernel_size,num_filt_1,
        num_filt_2,num_nrn_dl,num_nrn_ol,conv_strides ,
        pool_size_1,pool_size_2,pool_strides_1,pool_strides_2,epochs,dropout_rate,learning_rate,anm_det_thr )
        testingObj.run()
    else:
        trainModelObj = trainModel(dataPath,w ,p_w,n_features,kernel_size,num_filt_1,
        num_filt_2,num_nrn_dl,num_nrn_ol,conv_strides ,
        pool_size_1,pool_size_2,pool_strides_1,pool_strides_2,epochs,dropout_rate,learning_rate,anm_det_thr )
        trainModelObj.run()
