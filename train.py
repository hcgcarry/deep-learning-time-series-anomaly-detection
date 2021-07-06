# Things to be included
# 1. visualization tools for optimization
# 2. visualization tools for plotting actual and predicted sequence, and anomaly points
# 3. Computation cost calculation
# 4. Warnings / Errors?

import os
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
#from keras.optimizers import SGD

"""Generate model for predictor"""
class trainModel:
    def __init__(self,dataPath,w ,p_w,n_features,kernel_size,num_filt_1,
        num_filt_2,num_nrn_dl,num_nrn_ol,conv_strides ,
        pool_size_1,pool_size_2,pool_strides_1,pool_strides_2,epochs,dropout_rate,learning_rate,anm_det_thr ):
        """Hyperparameters"""
        self.dataPath = dataPath
        self.w = w 
        self.p_w = p_w
        self.n_features = n_features           # Univariate time series

        self.kernel_size = kernel_size          # Size of filter in conv layers
        self.num_filt_1 = num_filt_1 # Number of filters in first conv layer
        self.num_filt_2 = num_filt_2 # Number of filters in second conv layer
        self.num_nrn_dl = num_nrn_dl# Number of neurons in dense layer
        self.num_nrn_ol = num_nrn_ol        # Number of neurons in output layer

        self.conv_strides = conv_strides 
        self.pool_size_1 = pool_size_1          # Length of window of pooling layer 1
        self.pool_size_2 = pool_size_2          # Length of window of pooling layer 2
        self.pool_strides_1 = pool_strides_1       # Stride of window of pooling layer 1
        self.pool_strides_2 = pool_strides_2       # Stride of window of pooling layer 2

        self.epochs =epochs 
        self.dropout_rate = dropout_rate      # Dropout rate in the fully connected layer
        self.learning_rate = learning_rate  
        self.anm_det_thr = anm_det_thr       # Threshold for classifying anomaly (0.5~0.8)



    def run(self):
        #self.loadData('https://raw.githubusercontent.com/swlee23/Deep-Learning-Time-Series-Anomaly-Detection/master/data/sinewave.csv')
        self.loadData()
        self.buildModel()
        self.train()
    def runPredictMultiStep(self):
        #self.loadData('https://raw.githubusercontent.com/swlee23/Deep-Learning-Time-Series-Anomaly-Detection/master/data/sinewave.csv')
        self.loadData()
        self.buildModelMultiStep()
        self.train()

    def train(self):

        model_fit = self.model.fit(self.batch_sample,
                            self.batch_label,
                            epochs=self.epochs,
                            verbose=1)


        # save it to disk so we can load it back up anytime
        self.model.save_weights('model/model_predictWitdh_'+str(self.p_w)+'.h5')
    def loadData(self):
        """Data loading"""
        df= pd.read_csv(self.dataPath)
        # define input sequence
        '''
        raw_seq_1 = list(df['sinewave'])
        raw_seq_2 = list(df['sinewave'])
        for i in range(len(raw_seq_2)):
            raw_seq_2[(i+777)%len(raw_seq_2)]= raw_seq_1[i]
        raw_seq = np.stack((raw_seq_1,raw_seq_2),axis=1)
        with open("wavesine.csv","w") as f:
            for index in range(len(raw_seq)):
                print(raw_seq[index,0],",",raw_seq[index,1],file=f)
        '''
        raw_seq= df.to_numpy()

        for index in range(raw_seq.shape[1]):
            plt.figure(figsize=(100,10))
            plt.plot(raw_seq[:,index])
            plt.title('dimension_'+str(index))
            plt.ylabel('value')
            plt.xlabel('time')
            plt.legend(['target'], loc='upper right')
            plt.savefig("result/"+self.dataPath.split('/')[-2]+"/train_dim_"+str(index)+".png")

        # split into samples
        self.batch_sample, self.batch_label = split_sequence(raw_seq,self.w,self.p_w)

        print("batch_label shape",self.batch_label.shape)
        print("batch_label ",self.batch_label)
        self.batch_label = array([ item.flatten() for item in self.batch_label])
        print("batch_label shape",self.batch_label.shape)
        print("batch_label ",self.batch_label)

        # 2. reshape from [samples, timesteps] into [samples, timesteps, features]

        # need to convert batch into 3D tensor of the form [batch_size, input_seq_len, n_features]
        # 一個 batch 是所有的 windows
        self.batch_sample = self.batch_sample.reshape((self.batch_sample.shape[0], self.batch_sample.shape[1], self.n_features))
        # print("batch_sample shape",self.batch_sample.shape)
        # print("batch_label shape",self.batch_label.shape)
        # print("batch_label",self.batch_label)
    def buildModelMultiStep(self):
        self.model = Sequential()

        # Convolutional Layer #1
        # Computes 32 features using a 1D filter(kernel) of with w with ReLU activation. 
        # Padding is added to preserve width.
        # Input Tensor Shape: [batch_size, w, 1] ,batch_size = len(batch_sample)
        # Output Tensor Shape: [batch_size, w, num_filt_1] (num_filt_1 = 32 feature vectors)
        self.model.add(Conv1D(filters=self.num_filt_1,
                        kernel_size=self.kernel_size,
                        strides=self.conv_strides,
                        padding='valid',
                        activation='relu',
                        input_shape=(self.w, self.n_features)))

        # Pooling Layer #1
        # First max pooling layer with a filter of length 2 and stride of 2
        # Input Tensor Shape: [batch_size, w, num_filt_1]
        # Output Tensor Shape: [batch_size, 0.5 * w, num_filt_1]

        self.model.add(MaxPooling1D(pool_size=self.pool_size_1)) 
                            #  strides=pool_strides_1, 
                            #  padding='valid'))

        # Convolutional Layer #2
        # Computes 64 features using a 5x5 filter.
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, 0.5 * w, 32]
        # Output Tensor Shape: [batch_size, 0.5 * w, num_filt_1 * num_filt_2]
        self.model.add(Conv1D(filters=self.num_filt_2,
                        kernel_size=self.kernel_size,
                        strides=self.conv_strides,
                        padding='valid',
                        activation='relu'))

        # Max Pooling Layer #2
        # Second max pooling layer with a 2x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, 0.5 * w, num_filt_1 * num_filt_2]
        # Output Tensor Shape: [batch_size, 0.25 * w, num_filt_1 * num_filt_2]
        self.model.add(MaxPooling1D(pool_size=self.pool_size_2))
                            #  strides=pool_strides_2, 
                            #  padding='valid'
                
        # Flatten tensor into a batch of vectors
        # Input Tensor Shape: [batch_size, 0.25 * w, num_filt_1 * num_filt_2]
        # Output Tensor Shape: [batch_size, 0.25 * w * num_filt_1 * num_filt_2]
        self.model.add(Flatten())

        # Dense Layer (Output layer)
        # Densely connected layer with 1024 neurons
        # Input Tensor Shape: [batch_size, 0.25 * w * num_filt_1 * num_filt_2]
        # Output Tensor Shape: [batch_size, 1024]
        self.model.add(Dense(units=self.num_nrn_dl, activation='relu'))  

        # Dropout
        # Prevents overfitting in deep neural networks
        self.model.add(Dropout(self.dropout_rate))

        # Output layer
        # Input Tensor Shape: [batch_size, 1024]
        # Output Tensor Shape: [batch_size, p_w]
        self.model.add(Dense(units=self.n_features*self.p_w))

        # Summarize model structure
        self.model.summary()

        '''configure model'''
        self.model.compile(optimizer='adam', 
                    loss='mean_absolute_error')

        # sgd = keras.optimizers.SGD(lr=learning_rate, 
        #                          decay=1e-6, 
        #                          momentum=0.9, 
        #                          nesterov=True)
        # model.compile(optimizer='sgd', 
        #               loss='mean_absolute_error', 
        #               metrics=['accuracy'])


    def buildModel(self):
        self.model = Sequential()

        # Convolutional Layer #1
        # Computes 32 features using a 1D filter(kernel) of with w with ReLU activation. 
        # Padding is added to preserve width.
        # Input Tensor Shape: [batch_size, w, 1] ,batch_size = len(batch_sample)
        # Output Tensor Shape: [batch_size, w, num_filt_1] (num_filt_1 = 32 feature vectors)
        self.model.add(Conv1D(filters=self.num_filt_1,
                        kernel_size=self.kernel_size,
                        strides=self.conv_strides,
                        padding='valid',
                        activation='relu',
                        input_shape=(self.w, self.n_features)))

        # Pooling Layer #1
        # First max pooling layer with a filter of length 2 and stride of 2
        # Input Tensor Shape: [batch_size, w, num_filt_1]
        # Output Tensor Shape: [batch_size, 0.5 * w, num_filt_1]

        self.model.add(MaxPooling1D(pool_size=self.pool_size_1)) 
                            #  strides=pool_strides_1, 
                            #  padding='valid'))

        # Convolutional Layer #2
        # Computes 64 features using a 5x5 filter.
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, 0.5 * w, 32]
        # Output Tensor Shape: [batch_size, 0.5 * w, num_filt_1 * num_filt_2]
        self.model.add(Conv1D(filters=self.num_filt_2,
                        kernel_size=self.kernel_size,
                        strides=self.conv_strides,
                        padding='valid',
                        activation='relu'))

        # Max Pooling Layer #2
        # Second max pooling layer with a 2x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, 0.5 * w, num_filt_1 * num_filt_2]
        # Output Tensor Shape: [batch_size, 0.25 * w, num_filt_1 * num_filt_2]
        self.model.add(MaxPooling1D(pool_size=self.pool_size_2))
                            #  strides=pool_strides_2, 
                            #  padding='valid'
                
        # Flatten tensor into a batch of vectors
        # Input Tensor Shape: [batch_size, 0.25 * w, num_filt_1 * num_filt_2]
        # Output Tensor Shape: [batch_size, 0.25 * w * num_filt_1 * num_filt_2]
        self.model.add(Flatten())

        # Dense Layer (Output layer)
        # Densely connected layer with 1024 neurons
        # Input Tensor Shape: [batch_size, 0.25 * w * num_filt_1 * num_filt_2]
        # Output Tensor Shape: [batch_size, 1024]
        self.model.add(Dense(units=self.num_nrn_dl, activation='relu'))  

        # Dropout
        # Prevents overfitting in deep neural networks
        self.model.add(Dropout(self.dropout_rate))

        # Output layer
        # Input Tensor Shape: [batch_size, 1024]
        # Output Tensor Shape: [batch_size, p_w]
        self.model.add(Dense(units=self.n_features))

        # Summarize model structure
        self.model.summary()

        '''configure model'''
        self.model.compile(optimizer='adam', 
                    loss='mean_absolute_error')

        # sgd = keras.optimizers.SGD(lr=learning_rate, 
        #                          decay=1e-6, 
        #                          momentum=0.9, 
        #                          nesterov=True)
        # model.compile(optimizer='sgd', 
        #               loss='mean_absolute_error', 
        #               metrics=['accuracy'])

