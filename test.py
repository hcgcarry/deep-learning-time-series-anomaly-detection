# Things to be included
# 1. visualization tools for optimization
# 2. visualization tools for plotting actual and predicted sequence, and anomaly points
# 3. Computation cost calculation
# 4. Warnings / Errors?

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
from util import split_sequence
#from keras.optimizers import SGD

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1" #model will be trained on GPU 1



"""Data preprocessing"""
# split a univariate sequence into samples



"""Predicting random intervals (DeepAnT)"""
# Build model 
class testing:
    def __init__(self,w ,p_w,n_features,kernel_size,num_filt_1,
        num_filt_2,num_nrn_dl,num_nrn_ol,conv_strides ,
        pool_size_1,pool_size_2,pool_strides_1,pool_strides_2,epochs,dropout_rate,learning_rate,anm_det_thr ):
        """Hyperparameters"""
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
        self.loadData()
        self.loadModel()
        self.predict()

    def loadData(self):
        """Data loading"""
        df_sine = pd.read_csv('https://raw.githubusercontent.com/swlee23/Deep-Learning-Time-Series-Anomaly-Detection/master/data/sinewave.csv')
        raw_seq_1 = list(df_sine['sinewave'])
        raw_seq_2 = list(df_sine['sinewave'])
        for i in range(len(raw_seq_2)):
            raw_seq_2[(i+777)%len(raw_seq_2)]= raw_seq_1[i]
        self.raw_seq = np.stack((raw_seq_1,raw_seq_2),axis=1)

        plt.figure(figsize=(100,10))
        plt.plot(raw_seq_1)
        plt.plot(raw_seq_2)
        plt.title('sinewave')
        plt.ylabel('value')
        plt.xlabel('time')
        plt.legend(['input_seq','shifted_input_seq'], loc='upper right')
        plt.savefig("result/testcase.png")
    def loadModel(self):
        self.model = Sequential()
        self.model.add(Conv1D(filters=self.num_filt_1,
                        kernel_size=self.kernel_size,
                        strides=self.conv_strides,
                        padding='valid',
                        activation='relu',
                        input_shape=(self.w,self. n_features)))
        self.model.add(MaxPooling1D(pool_size=self.pool_size_1)) 
        self.model.add(Conv1D(filters=self.num_filt_2,
                        kernel_size=self.kernel_size,
                        strides=self.conv_strides,
                        padding='valid',
                        activation='relu'))
        self.model.add(MaxPooling1D(pool_size=self.pool_size_2))
        self.model.add(Flatten())
        self.model.add(Dense(units=self.num_nrn_dl, activation='relu')) 
        self.model.add(Dropout(self.dropout_rate))
        self.model.add(Dense(units=self.n_features))

        self.model.load_weights('model/sinwave_DeepAnT_1.h5')
    def predict(self):
        batch_sample, batch_label = split_sequence(self.raw_seq,self.w)
        # Predict the next time stampes of the sampled sequence
        predictResult= self.model.predict(batch_sample, verbose=1)

        # Print our model's predictions.
        print("predict:",predictResult)
        print("predict:",predictResult.shape)

        # Check our predictions against the ground truths.
        print("ground true",batch_label) # [7, 2, 1, 0, 4]
        print("ground true",batch_label.shape) # [7, 2, 1, 0, 4]``

        
        raw_seq_1 = self.raw_seq[:,0]
        raw_seq_2 = self.raw_seq[:,1]
        plt.figure(figsize=(100,10))
        
        plt.plot(batch_label[:,0])
        plt.plot(batch_label[:,1])
        plt.plot(predictResult[:,0])
        plt.plot(predictResult[:,1])
        plt.title('sinewave')
        plt.ylabel('value')
        plt.xlabel('time')
        plt.legend(['label_dim0','label_dim1','predict_dim0','predict_dim1'], loc='upper right')
        plt.savefig("result/testResult.png")




    '''
    def testold(self):
        model = Sequential()
        model.add(Conv1D(filters=num_filt_1,
                        kernel_size=kernel_size,
                        strides=conv_strides,
                        padding='valid',
                        activation='relu',
                        input_shape=(w, n_features)))
        model.add(MaxPooling1D(pool_size=pool_size_1)) 
        model.add(Conv1D(filters=num_filt_2,
                        kernel_size=kernel_size,
                        strides=conv_strides,
                        padding='valid',
                        activation='relu'))
        model.add(MaxPooling1D(pool_size=pool_size_2))
        model.add(Flatten())
        model.add(Dense(units=num_nrn_dl, activation='relu')) 
        model.add(Dropout(dropout_rate))
        model.add(Dense(units=num_nrn_ol))

        # Load the model's saved weights.
        model.load_weights('model/sinwave_DeepAnT_1.h5')
                
        # Sample a portion of the raw_seq randomly
        # 1. Choose 
        ran_ix = random.randint(1,len(raw_seq) - w - p_w)
        input_seq = array(raw_seq[ran_ix : ran_ix + w])
        target_seq = array(raw_seq[ran_ix + w ])
        input_seq = input_seq.reshape((1, w, n_features))

        # Predict the next time stampes of the sampled sequence
        yhat = model.predict(input_seq, verbose=1)

        # Print our model's predictions.
        print("predict:",yhat)
        print("predict:",yhat.shape)

        # Check our predictions against the ground truths.
        print("ground true",target_seq) # [7, 2, 1, 0, 4]
        print("ground true",target_seq.shape) # [7, 2, 1, 0, 4]

        """Predicting future sequence (DeepAnT)"""
        # Build model 
        model = Sequential()
        model.add(Conv1D(filters=num_filt_1,
                        kernel_size=kernel_size,
                        strides=conv_strides,
                        padding='valid',
                        activation='relu',
                        input_shape=(w, n_features)))
        model.add(MaxPooling1D(pool_size=pool_size_1)) 
        model.add(Conv1D(filters=num_filt_2,
                        kernel_size=kernel_size,
                        strides=conv_strides,
                        padding='valid',
                        activation='relu'))
        model.add(MaxPooling1D(pool_size=pool_size_2))
        model.add(Flatten())
        model.add(Dense(units=num_nrn_dl, activation='relu')) 
        model.add(Dropout(dropout_rate))
        model.add(Dense(units=num_nrn_ol))

        # Load the model's saved weights.
        model.load_weights('model/sinwave_DeepAnT_1.h5')
                
            
        raw_seq = list(df_sine['sinewave'])
        endix = len(raw_seq) - w - p_w
        input_seq = array(raw_seq[endix:endix+w])
        target_seq = array(raw_seq[endix+w:endix+w+p_w]) 
        input_seq = input_seq.reshape((1, w, n_features))

        # Predict the next time stampes of the sampled sequence
        predicted_seq = model.predict(input_seq, verbose=1)

        # Print our model's predictions.
        print(predicted_seq)

        # Check our predictions against the ground truths.
        print(target_seq) # [7, 2, 1, 0, 4]

        in_seq = df_sine['sinewave'][endix:endix+w]
        tar_seq = df_sine['sinewave'][endix+w:endix+w+p_w]
        predicted_seq = predicted_seq.reshape((p_w))
        d = {'time': df_sine['time'][endix+w:endix+w+p_w], 'values': predicted_seq}
        df_sine_pre = pd.DataFrame(data=d)
        pre_seq = df_sine_pre['values']

        fig_predict = plt.figure(figsize=(100,10))
        plt.plot(in_seq)
        plt.plot(tar_seq)
        plt.plot(pre_seq)
        plt.title('sinewave prediction')
        plt.ylabel('value')
        plt.xlabel('time')
        plt.legend(['input_seq', 'target_seq', 'pre_seq'], loc='upper right')
        axes = plt.gca()
        axes.set_xlim([endix,endix+w+p_w])
        fig_predict.savefig('result/predicted_sequence.png')
        plt.show()    
    '''

    