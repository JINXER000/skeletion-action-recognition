# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 10:19:45 2018

@author: Jinxe
"""

import numpy as np
np.random.seed(0)
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
np.random.seed(1)
import matplotlib.pyplot as plt
import  extract_key as ek
from emo_utils import *



    
def key2action(input_shape):
    keyset=Input(shape=input_shape,dtype='float32')
    #keyset=Input(shape=(27,50),dtype='float32')
    X = LSTM(128, return_sequences=True)(keyset)
    X=Dropout(0.5)(X)
    X = LSTM(128)(X)
    X = Dropout(0.5)(X)
    X = Dense(2)(X)
    X = Activation('softmax')(X)
    model = Model(inputs=keyset, output=X)
    return model

if __name__=='__main__':    
    #load data
    targetDir=r'F:\CV_WS\machine_learning\openpose-1.4.0\openpose-1.4.0\out_json'
    X_data,max_frame,Y_data=ek.process_all_sec(targetDir)
    #shuffle index
    np.random.seed(1024)
    data_index=[i for i in range( X_data.shape[0])]
    np.random.shuffle(data_index)
    X_data_sf=X_data[data_index]
    Y_data_sf=Y_data[data_index]
    splitpoint = int(round(len(data_index) * 0.8))
    (X_train, X_val) =(X_data_sf[0:splitpoint],X_data_sf[splitpoint:])
    (Y_train, Y_val)=(Y_data_sf[0:splitpoint],Y_data_sf[splitpoint:])
    Y_oh_train=convert_to_one_hot(Y_train,C=2)
    #train model
    model=key2action((max_frame,50,))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history=model.fit(X_train, Y_oh_train, epochs = 50, batch_size = 16, shuffle=True)

    
    ##plt
    fig = plt.figure()


    
    #plt.plot(history.history['val_acc'])
    
    plt.title('model accuracy')
    
    plt.ylabel('accuracy')
    
    plt.xlabel('epoch')
    plt.plot(history.history['acc'])
    
    #plt.legend(['train', 'test'], loc='upper left')
    
    plt.plot(history.history['loss'])
    
    #plt.plot(history.history['val_loss'])
    
    plt.title('model loss')
    
    plt.ylabel('loss')
    
    plt.xlabel('epoch')
    
    #plt.legend(['train', 'test'], loc='lower left')
    
    fig.savefig('performance.png')
    
    Y_val_oh=convert_to_one_hot(Y_val,C=2)
    loss,acc=model.evaluate(X_val,Y_val_oh)
    
    secDir=r'boxing_55'
    x_test1=pass_x_once(targetDir,secDir)
    pred=model.predict(x_test1)
    num = np.argmax(pred)


    
