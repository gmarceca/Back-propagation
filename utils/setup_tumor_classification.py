import h5py
import numpy as np
from random import shuffle
from math import ceil
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data_RN ():

    dataset = pd.read_csv('./tp1_ej1_training.csv')
    y = dataset.iloc[:,0].values
    X = dataset.iloc[:,1:dataset.shape[1]].values

    # 1 = 'M', 0 = 'B'
    y = (y=='M').astype(int)

    # split the data in training, validation and testing
    X_ori, X_test_ori, y_ori, y_test_ori = train_test_split(X, y, test_size=0.2,random_state=123,stratify=y) # 20% testing
    X_train_ori, X_val_ori, y_train_ori, y_val_ori = train_test_split(X_ori, y_ori, test_size=0.25,random_state=123,stratify=y_ori) #80%x25% = 20% validation and 80%x75% = 60% training 


    # Normalize the data
    mu = np.mean(X_train_ori,axis=0)
    std = np.std(X_train_ori,axis=0)


    train_x_normalized = (X_train_ori - mu)/std
    val_x_normalized = (X_val_ori - mu)/std
    test_x_normalized = (X_test_ori - mu)/std

    train_x = train_x_normalized.T
    val_x = val_x_normalized.T
    test_x = test_x_normalized.T    
    train_y = y_train_ori.reshape(1,y_train_ori.shape[0])
    val_y = y_val_ori.reshape(1,y_val_ori.shape[0])
    test_y = y_test_ori.reshape(1,y_test_ori.shape[0])


    return train_x, train_y, val_x, val_y, test_x, test_y
