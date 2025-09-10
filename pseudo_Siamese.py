#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 1 19:22:35 2020

@author: liuhongyi
"""




import pandas as pd
#import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Input, Dense
from keras.layers.core import Dropout, Lambda
from keras.optimizers import Adam
#from keras.layers import BatchNormalization
from tensorflow.python.keras.models import Model
#from keras import backend as K
#import theano.tensor as T

#from pandas import DataFrame
#from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
#from util import ManDist
from keras.models import load_model
from sklearn.metrics import r2_score
import sys


# flags = tf.app.flags
# FLAGS = flags.FLAGS
# flags.DEFINE_float('lr', 0.01, 'Initial learning rate.')
# flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
# flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
# flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
#                      'Must divide evenly into the dataset sizes.')
# flags.DEFINE_integer('epoch', 50, 'Epoch')
# flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
# flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
#                      'for unit testing.')


#language, year, model, jobid = sys.argv[1:5]

n_epoch = 50
batch_size = 16

ln_hidden = 50
rn_hidden = 200

dropout_rate = 0.3
learning_rate = 0.0001

# def get_batch(loc, batch_size, mode):
#   if mode == 'train':
#     batch_id = train_id_dataset[loc:loc+batch_size]
#     ehr = np.array(train_ehr_dataset)[batch_id]
#     demo = np.array(train_demo_dataset)[batch_id]
#     cur_trial = train_trial_dataset[loc:loc+batch_size]
#     batch_label = train_label_dataset[loc:loc+batch_size]
#   elif mode == 'valid':
#     batch_id = valid_id_dataset[loc:loc+batch_size]
#     ehr = np.array(valid_ehr_dataset)[batch_id]
#     demo = np.array(valid_demo_dataset)[batch_id]
#     cur_trial = valid_trial_dataset[loc:loc+batch_size]
#     batch_label = valid_label_dataset[loc:loc+batch_size]
#   else:
#     batch_id = test_id_dataset[loc:loc+batch_size]
#     ehr = np.array(test_ehr_dataset)[batch_id]
#     demo = np.array(test_demo_dataset)[batch_id]
#     cur_trial = test_trial_dataset[loc:loc+batch_size]
#     batch_label = test_label_dataset[loc:loc+batch_size]


def train_test_split(data, d1, d2):
    x_train, y_train = data.drop(columns=['eret']).loc[data.date < d1,:].drop(columns='date'), data.loc[data.date < d1,'eret']
    x_validate, y_validate = data.drop(columns=['eret']).loc[(data.date < d2) & (data.date >= d1),:].drop(columns='date'), data.loc[(data.date < d2) & (data.date >= d1),'eret']
    x_test, y_test = data.drop(columns=['eret']).loc[data.date >= d2,:].drop(columns='date'), data.loc[ data.date >= d2,'eret']
    return x_train, y_train, x_validate, y_validate, x_test, y_test


def reshape_data(train,validate, test):
    #Frame as supervised learning and drop all time t columns except PM2.5

    # split into train and test sets
    train_X = train.values
    validate_X = validate.values
    test_X = test.values

    # reshape input to be 3D [samples, timesteps, features]
    x_train = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    x_validate = validate_X.reshape((validate_X.shape[0], 1, validate_X.shape[1]))
    x_test = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    return x_train,x_validate, x_test


def data_processing(full_chars, common_factors, d1, d2):
    # Split to training, validation, and test set
    # unique stock
    firms = list(set(full_chars.firm))

    # loop each station and collect train, validation, and test data 

    X_train, X_validate, X_test = list(), list(), list()
    Z_train, Z_validate, Z_test = list(), list(), list()
    Y_train, Y_validate, Y_test = list(), list(), list()

    for i in range(0,len(firms)):
        data = full_chars[full_chars['firm'] == firms[i]]
        dt = data[['date']]
        z = pd.merge(common_factors,dt, how='inner', on='date')
        ztrain = z.loc[z.date < d1,:].drop(columns='date') 
        zvalidate = z.loc[(z.date < d2) & (z.date >= d1),:].drop(columns='date')
        ztest = z.loc[z.date >= d2,:].drop(columns='date')
    
    
        xtrain, ytrain, xvalidate, yvalidate, xtest, ytest = train_test_split(data, d1, d2)
    
        X_train.append(xtrain)
        X_validate.append(xvalidate)
        X_test.append(xtest)
            
        Y_train.append(ytrain)
        Y_validate.append(yvalidate)
        Y_test.append(ytest)
        
        Z_train.append(ztrain)
        Z_validate.append(zvalidate)
        Z_test.append(ztest)
    
    # concat each train data from each station 
    X_train = pd.concat(X_train)
    Y_train = pd.DataFrame(pd.concat(Y_train))
    # concat each validate data from each station 

    X_validate = pd.concat(X_validate)
    Y_validate = pd.DataFrame(pd.concat(Y_validate))

    X_test=pd.concat(X_test)
    Y_test=pd.DataFrame(pd.concat(Y_test))


    Z_train = pd.concat(Z_train)
    Z_validate = pd.concat(Z_validate)
    Z_test=pd.concat(Z_test)
    
    x_train,x_validate, x_test = reshape_data(X_train, X_validate,X_test)
    z_train,z_validate, z_test = reshape_data(Z_train,Z_validate, Z_test)
    
    y_train, y_validate, y_test = Y_train.values, Y_validate.values, Y_test.values

    
    return x_train, y_train, z_train, x_validate, y_validate, z_validate, x_test, y_test, z_test

 
# model.add(Flatten())


def Siamese_lstm(left, right, y, batch_size, nb_epoch, ln_hidden, rn_hidden, dropout_rate, learning_rate):
    
    lmodel = Sequential()
    lmodel.add(LSTM(ln_hidden, activation='relu', input_shape=(left.shape[1], left.shape[2]), return_sequences=True))
    lmodel.add(Dropout(dropout_rate))
    lmodel.add(Dense(1))
    
    rmodel = Sequential()
    rmodel.add(LSTM(rn_hidden, activation='relu', input_shape=(right.shape[1], right.shape[2]), return_sequences=True))
    rmodel.add(Dropout(dropout_rate))
    rmodel.add(Dense(1))
    
    left_input = Input(shape= (left.shape[1], left.shape[2]), dtype='float32')
    right_input = Input(shape=(right.shape[1], right.shape[2]), dtype='float32')
    
    encoded_l = lmodel(left_input)
    encoded_r = rmodel(right_input)
    L1_layer = Lambda(lambda tensors:(tensors[0] * tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    prediction = Dense(1)(L1_distance)
    
    model = Model(inputs=[left_input,right_input],outputs = prediction)
    optimizer = Adam(lr = learning_rate)
    model.compile(loss='mse',optimizer=optimizer, metrics=['accuracy'])   
    model.fit( [left, right], y, epochs = nb_epoch, batch_size=batch_size, verbose=2)
    
    return model


    
    


#Full_CSV = './sample.csv'
Full_CSV = './full.csv'
CF_CSV = './common_factors.csv' # remove UMCSENTx TWEXAFEGSMTHx ACOGNO ANDENOx and other 12 firm chars
# Load training set
full_df = pd.read_csv(Full_CSV, parse_dates=['DATE'],error_bad_lines=False, warn_bad_lines=False).drop(columns=['Unnamed: 0'])



### firm characteristics
# normalized the data 
scaler1 = MinMaxScaler(feature_range=(0, 1))
scaled_full = scaler1.fit_transform(full_df.drop(columns=['permno','DATE','eret']))
# create datefrane fir scaled data
scaled_full_df=pd.DataFrame(data=scaled_full,columns = full_df.drop(columns=['permno','DATE','eret']).columns)
full_chars = full_df.copy()
full_chars[scaled_full_df.columns] = np.array(scaled_full_df)
full_chars['firm'] = list(full_chars.permno)
full_chars['date'] = list(full_chars.DATE)
full_chars = full_chars.set_index(['permno','DATE'])

#full_chars.dropna(inplace=True)
#full_chars.head()

### average firm characteristics and macroeconomic variable
scaler2 = MinMaxScaler(feature_range=(0, 1))
cf_df = pd.read_csv(CF_CSV, parse_dates=['date'],error_bad_lines=False, warn_bad_lines=False)
scaled_cf = scaler2.fit_transform(cf_df.drop(columns=['date']))
scaled_cf_df=pd.DataFrame(data=scaled_cf,columns = cf_df.drop(columns=['date']).columns)
common_factors = cf_df.copy()

common_factors[scaled_cf_df.columns] = np.array(scaled_cf_df)
#common_factors.head()

# sample = full_df[full_df.permno < 15000]
# sample.to_csv('sample.csv')




date1, date2 = '2006-01-01', '2011-01-01'

x_train, y_train, z_train, x_validate, y_validate, z_validate, x_test, y_test, z_test = data_processing(full_chars, common_factors, date1, date2)

model = Siamese_lstm(x_train, z_train, y_train, batch_size, n_epoch, ln_hidden, rn_hidden, dropout_rate, learning_rate)
model_path = './SLSTM' + '_'+ str(n_epoch) + '_'+ str(batch_size) + '_'+ str(ln_hidden) + '_'+ str(rn_hidden)+ '_'+ str(dropout_rate) + '_'+ str(learning_rate) + '.h5'
model.save(model_path)

# from time import time
#training_start_time = time()

#training_end_time = time()
# print("Training time finished.\n%d epochs in %12.2f" % (n_epoch,
#                                                         training_end_time - training_start_time))

#model.save('./SiameseLSTM.h5')

#model = load_model('SLSTM_sample.h5')
#model.summary()



# score = model.evaluate([x_test,z_test], y_test, verbose=0)
# ytrain_hat = model.predict([x_train,z_train])
# yvalidat_hat = model.predict([x_validate,z_validate])
# ytest_hat = model.predict([x_test, z_test])

# # R2_TS
# r2 = r2_score(y_train, np.sum(ytrain_hat,axis=1))
# # new_Y = Y_train.reset_index(inplace=True)
# # mispricing errrors
# #Y_train.reset_index(inplace=True)
# alpha_train_df = pd.concat([pd.DataFrame(y_train), pd.DataFrame(np.sum(ytrain_hat, axis=1))], axis=1)
# alpha_train_df.rename(columns={0: "yhat"},inplace=True)
# alpha_train_df['res'] = alpha_train_df.eret - alpha_train_df.yhat
# train_alpha_hat = alpha_train_df.groupby('permno')['res'].mean().pow(2).mean()

# predict 

#Xtrain_hat = pd.concat([pd.DataFrame(np.sum(x_train, axis=1)[:,0:-1]), pd.DataFrame(np.sum(ytrain_hat, axis=1))], axis=1)
