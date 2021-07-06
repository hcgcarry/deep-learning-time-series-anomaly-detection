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
from util import split_sequence,split_sequence_multiStep
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score

#from keras.optimizers import SGD

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1" #model will be trained on GPU 1



"""Data preprocessing"""
# split a univariate sequence into samples



"""Predicting random intervals (DeepAnT)"""
# Build model 
class testing:
    def __init__(self,dataPath,w ,p_w,n_features,kernel_size,num_filt_1,
        num_filt_2,num_nrn_dl,num_nrn_ol,conv_strides ,
        pool_size_1,pool_size_2,pool_strides_1,pool_strides_2,epochs,dropout_rate,learning_rate,anm_det_thr ):
        """Hyperparameters"""
        self.dataPath=dataPath
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
        self.result_save_path = "result/"+self.dataPath.split('/')[-2]+"/"

    def run(self):
        self.loadData()
        self.loadModel()
        self.predict()
    def runPredictMultiStep(self):
        self.loadData()
        self.loadModelMultiStep()
        self.predictMultiStep()


    def loadData(self):
        """Data loading"""
        df= pd.read_csv(self.dataPath)
        self.raw_seq= df.to_numpy()

    def caculateMeanAndVariance(self):
        mean = np.mean(self.raw_seq,0)
        variance = np.std(self.raw_seq,0)
        print("mean",mean  )
        print("variance ",variance)
        return (mean,variance)


    def loadModelMultiStep(self):
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
        self.model.add(Dropout(0))
        self.model.add(Dense(units=self.n_features*self.p_w))

        self.model.load_weights('model/model_predictWitdh_'+str(self.p_w)+'.h5')
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
        self.model.add(Dropout(0))
        self.model.add(Dense(units=self.n_features))

        self.model.load_weights('model/model_predictWitdh_'+self.p_w+'.h5')
    def predictMultiStep(self):
        mean,std= self.caculateMeanAndVariance()
        batch_sample, batch_label = split_sequence_multiStep(self.raw_seq,self.w,self.p_w)
        batch_label= batch_label.reshape((-1,self.n_features))
        # Predict the next time stampes of the sampled sequence
        predictResult= self.model.predict(batch_sample, verbose=1)
        predictResult = predictResult.reshape((-1,self.n_features))
        
        total_anomaly_list,total_mean_absolute_error_list,mean_absolute_error_list,anomaly_list = self.caculateError(predictResult,batch_label)

        
        ## print result
        with open(self.result_save_path+"anomaly_list.txt","w") as f:
            f.write("anomaly_list:\n")
            for dimIndex in range(self.n_features):
                f.write("dim:"+str(dimIndex)+" ")
                f.write(' '.join('{},{:.2f}'.format(x[0],x[1]) for x in anomaly_list[dimIndex]))
                f.write("\n")

            f.write("total_anomaly_list:\n")
            f.write(' '.join('{},{:.2f}'.format(x[0],x[1]) for x in total_anomaly_list))

        
        print("total_mean_absolute_erro_list:",total_mean_absolute_error_list)
        print("total error sum:",np.sum((total_mean_absolute_error_list)))
        print("total error mean:",np.mean(total_mean_absolute_error_list))
        print("total error std:",np.std(total_mean_absolute_error_list))

        for index in range(self.raw_seq.shape[1]):
            plt.figure(figsize=(100,10))
            plt.plot(batch_label[:,index])
            plt.plot(predictResult[:,index])
            plt.plot(mean_absolute_error_list[:,index],'r')
            plt.title('dimension_'+str(index))
            plt.ylabel('value')
            plt.xlabel('time')
            plt.xticks([item[0] for item in anomaly_list[index]])
            plt.legend(['target','predict','error'], loc='upper right')
            plt.savefig("result/"+self.dataPath.split('/')[-2]+"/test_dim_"+str(index)+".png")

        ## total mean absolute error fig
        plt.figure(figsize=(100,10))
        plt.plot(total_mean_absolute_error_list,'r')
        plt.title("total_mean_absolute_error")
        plt.ylabel('value')
        plt.xlabel('time')
        plt.xticks([item[0] for item in total_anomaly_list])
        plt.legend(['error'], loc='upper right')
        plt.savefig("result/"+self.dataPath.split('/')[-2]+"/test_total_mean_absolute_error.png")

    def caculateError(self,predictResult,batch_label):
        ################ error
        mean,std= self.caculateMeanAndVariance()
        mean_absolute_error_list = np.zeros(predictResult.shape)
        total_mean_absolute_error_list = np.zeros(predictResult.shape[0])
        
        anomaly_threshold = 0.7
        anomaly_list = [ [] for i in range(predictResult.shape[1])]
        total_anomaly_list = []
        for rowIndex in range(predictResult.shape[0]):
            sum =0
            for dimIndex in range(predictResult.shape[1]):
                curValue =abs(predictResult[rowIndex,dimIndex] - batch_label[rowIndex,dimIndex])
                mean_absolute_error_list[rowIndex,dimIndex] =curValue 
                sum+=abs((curValue - mean[dimIndex])/std[dimIndex])
                if (curValue-mean[dimIndex])/std[dimIndex] > anomaly_threshold:
                    anomaly_list[dimIndex].append((rowIndex,curValue))
            total_mean_absolute_error_list[rowIndex] = (sum/predictResult.shape[1])
            if total_mean_absolute_error_list[rowIndex] > anomaly_threshold:
                total_anomaly_list.append((rowIndex,total_mean_absolute_error_list[rowIndex]))

        return (total_anomaly_list,total_mean_absolute_error_list,mean_absolute_error_list,anomaly_list )
        ####### error end

    def predict(self):
        mean,std= self.caculateMeanAndVariance()
        batch_sample, batch_label = split_sequence(self.raw_seq,self.w,self.p_w)
        # Predict the next time stampes of the sampled sequence
        predictResult= self.model.predict(batch_sample, verbose=1)
        print("predictResult shape",predictResult.shape)
        
        mean_absolute_error_list = np.zeros(predictResult.shape)
        total_mean_absolute_error_list = np.zeros(predictResult.shape[0])
        
        anomaly_threshold = 0.7
        anomaly_list = [ [] for i in range(predictResult.shape[1])]
        total_anomaly_list = []
        for rowIndex in range(predictResult.shape[0]):
            sum =0
            for dimIndex in range(predictResult.shape[1]):
                curValue =abs(predictResult[rowIndex,dimIndex] - batch_label[rowIndex,dimIndex])
                mean_absolute_error_list[rowIndex,dimIndex] =curValue 
                sum+=(curValue - mean[dimIndex])/std[dimIndex]
                if (curValue-mean[dimIndex])/std[dimIndex] > anomaly_threshold:
                    anomaly_list[dimIndex].append((rowIndex,curValue))
            total_mean_absolute_error_list[rowIndex] = (sum/predictResult.shape[1])
            if total_mean_absolute_error_list[rowIndex] > anomaly_threshold:
                total_anomaly_list.append((rowIndex,total_mean_absolute_error_list[rowIndex]))

        
        with open(self.result_save_path+"anomaly_list.txt","w") as f:
            f.write("anomaly_list:\n")
            for dimIndex in range(self.n_features):
                f.write("dim:"+str(dimIndex)+" ")
                f.write(' '.join('{},{:.2f}'.format(x[0],x[1]) for x in anomaly_list[dimIndex]))
                f.write("\n")

            f.write("total_anomaly_list:\n")
            f.write(' '.join('{},{:.2f}'.format(x[0],x[1]) for x in total_anomaly_list))

        
        print("total error absolute sum:",np.sum(np.absolute(total_mean_absolute_error_list)))
        print("total error sum:",np.sum(total_mean_absolute_error_list))
        print("total error mean:",np.mean(total_mean_absolute_error_list))
        print("total error std:",np.std(total_mean_absolute_error_list))

        for index in range(self.raw_seq.shape[1]):
            plt.figure(figsize=(100,10))
            plt.plot(batch_label[:,index])
            plt.plot(predictResult[:,index])
            plt.plot(mean_absolute_error_list[:,index],'r')
            plt.title('dimension_'+str(index))
            plt.ylabel('value')
            plt.xlabel('time')
            plt.xticks([item[0] for item in anomaly_list[index]])
            plt.legend(['target','predict','error'], loc='upper right')
            plt.savefig("result/"+self.dataPath.split('/')[-2]+"/test_dim_"+str(index)+".png")

        ## total mean absolute error fig
        plt.figure(figsize=(100,10))
        plt.plot(total_mean_absolute_error_list,'r')
        plt.title("total_mean_absolute_error")
        plt.ylabel('value')
        plt.xlabel('time')
        plt.xticks([item[0] for item in total_anomaly_list])
        plt.legend(['error'], loc='upper right')
        plt.savefig("result/"+self.dataPath.split('/')[-2]+"/test_total_mean_absolute_error.png")


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

    