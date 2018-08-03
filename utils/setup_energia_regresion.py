import h5py
import numpy as np
from random import shuffle
from math import ceil
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data_RN ():

    dataset = pd.read_csv('./tp1_ej2_training.csv')
    y = dataset.iloc[:,-2:].values
    X = dataset.iloc[:,0:(dataset.shape[1]-2)].values

    # split the data in training, validation and testing
    #X_ori, X_test_ori, y_ori, y_test_ori = train_test_split(X, y, test_size=0.2,random_state=123) # 20% testing
    #X_train_ori, X_val_ori, y_train_ori, y_val_ori = train_test_split(X_ori, y_ori, test_size=0.25,random_state=123) #80%x25% = 20% validation and 80%x75% = 60% training

    X_train_ori, X_test_ori, y_train_ori, y_test_ori = train_test_split(X, y, test_size=0.1,random_state=123)

    # Normalize the data
    mu = np.mean(X_train_ori,axis=0)
    std = np.std(X_train_ori,axis=0)


    train_x_normalized = (X_train_ori - mu)/std
    test_x_normalized = (X_test_ori - mu)/std
#    val_x_normalized = (X_val_ori - mu)/std

    train_x = train_x_normalized.T
#    val_x = val_x_normalized.T
    test_x = test_x_normalized.T    
    train_y = y_train_ori.T
    test_y = y_test_ori.T
#    val_y = y_val_ori.T

    return train_x, train_y, test_x, test_y

def load_data_Cat_vs_Dog ():

    hdf5_path = '/Volumes/SAMSUNG/Cat-vs-Dog/dataset_64px64p_reduced.hdf5'
    subtract_mean = False
    # open the hdf5 file
    hdf5_file = h5py.File(hdf5_path, "r")
    # subtract the training mean
    if subtract_mean:
        mm = hdf5_file["train_mean"][0, ...]
        mm = mm[np.newaxis, ...]
    # Total number of samples
    train_x_orig = hdf5_file["train_img"][()]
    train_y_ori = hdf5_file["train_labels"][()]
    test_x_orig = hdf5_file["test_img"][()]
    test_y_ori = hdf5_file["test_labels"][()]
    
    train_y = np.expand_dims(train_y_ori, axis=1).T
    test_y = np.expand_dims(test_y_ori, axis=1).T
    
    hdf5_file.close()

    return train_x_orig, train_y, test_x_orig, test_y


def load_dataset_Cat_vs_nonCat():
    train_dataset = h5py.File('/Volumes/SAMSUNG/Cats-vs-nonCats/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('/Volumes/SAMSUNG/Cats-vs-nonCats/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

#print(type(train_x_orig))
#print(train_x_orig.shape)
#print(train_y.shape)

#train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
#print(train_x_flatten.shape)

#print(train_y.shape[0])
#print(test_x_orig.shape[0])
#print(test_y.shape[0])

#batch_size = 10
#nb_class = 2
#
## create list of batches to shuffle the data
#batches_list = list(range(int(ceil(float(data_num) / batch_size))))
##shuffle(batches_list)
## loop over batches
#for n, i in enumerate(batches_list):
#    i_s = i * batch_size  # index of the first image in this batch
#    i_e = min([(i + 1) * batch_size, data_num])  # index of the last image in this batch
#    
#    # read batch images and remove training mean
#    images = hdf5_file["train_img"][i_s:i_e, ...]
#
#    if subtract_mean:
#        images -= mm
#    # read labels and convert to one hot encoding
#    labels = hdf5_file["train_labels"][i_s:i_e]
#    labels_one_hot = np.zeros((batch_size, nb_class))
#    labels_one_hot[np.arange(batch_size), labels] = 1
#    print(n+1, '/', len(batches_list))
#    print(labels[0], labels_one_hot[0, :])
#    plt.imshow(images[0])
#    plt.savefig('test.pdf')
#    if n == 0:  # break after 5 batches
#        break
#hdf5_file.close()
