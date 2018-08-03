# http://machinelearninguru.com/deep_learning/data_preparation/hdf5/hdf5.html
from random import shuffle
import glob
import numpy as np
import h5py
import cv2

shuffle_data = True  # shuffle the addresses before saving
hdf5_path = '/Volumes/SAMSUNG/Cat-vs-Dog/dataset_64px64p_reduced.hdf5'  # address to where you want to save the hdf5 file
cat_dog_train_path = '/Volumes/SAMSUNG/Cat-vs-Dog/train/*.jpg'

# read addresses and labels from the 'train' folder
addrs = glob.glob(cat_dog_train_path)
labels = [0 if 'cat' in addr else 1 for addr in addrs]  # 0 = Cat, 1 = Dog

# to shuffle data
if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)

#print("addrs: ",addrs[0])

# Divide the data into 80% train, 20% test
train_addrs = addrs[0:int(0.8*len(addrs))]
train_labels = labels[0:int(0.8*len(labels))]
#val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
#val_labels = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]
test_addrs = addrs[int(0.8*len(addrs)):]
test_labels = labels[int(0.8*len(labels)):]

data_order = 'tf'  # 'th' for Theano, 'tf' for Tensorflow
# check the order of data and chose proper data shape to save images
if data_order == 'th':
    train_shape = (len(train_addrs), 3, 64, 64)
    #val_shape = (len(val_addrs), 3, 224, 224)
    test_shape = (len(test_addrs), 3, 64, 64)
elif data_order == 'tf':
    train_shape = (len(train_addrs), 64, 64, 3)
    #val_shape = (len(val_addrs), 224, 224, 3)
    test_shape = (len(test_addrs), 64, 64, 3)

# open a hdf5 file and create earrays
hdf5_file = h5py.File(hdf5_path, mode='w')

hdf5_file.create_dataset("train_img", train_shape, np.uint8)
#hdf5_file.create_dataset("val_img", val_shape, np.int8)
hdf5_file.create_dataset("test_img", test_shape, np.uint8)
hdf5_file.create_dataset("train_mean", train_shape[1:], np.float32)
hdf5_file.create_dataset("train_labels", (len(train_addrs),), np.int8)
hdf5_file["train_labels"][...] = train_labels
#hdf5_file.create_dataset("val_labels", (len(val_addrs),), np.int8)
#hdf5_file["val_labels"][...] = val_labels
hdf5_file.create_dataset("test_labels", (len(test_addrs),), np.int8)
hdf5_file["test_labels"][...] = test_labels

# a numpy array to save the mean of the images
mean = np.zeros(train_shape[1:], np.float32)

train_size = 209
test_size = 50

# loop over train addresses
for i in range(train_size):
    # print how many images are saved every 1000 images
    if i % 10 == 0 and i > 1:
        print('Train data: {}/{}'.format(i, len(train_addrs)))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = train_addrs[i]
    img = cv2.imread(addr)
    #img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # add any image pre-processing here
    # if the data order is Theano, axis orders should change
    if data_order == 'th':
        img = np.rollaxis(img, 2)
    # save the image and calculate the mean so far
    hdf5_file["train_img"][i, ...] = img[None]
    mean += img / float(len(train_labels))

    pass

# loop over test addresses
for i in range(test_size):
    # print how many images are saved every 1000 images
    if i % 10 == 0 and i > 1:
        print('Test data: {}/{}'.format(i, len(test_addrs)))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = test_addrs[i]
    img = cv2.imread(addr)
    #img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # add any image pre-processing here
    # if the data order is Theano, axis orders should change
    if data_order == 'th':
        img = np.rollaxis(img, 2)
    # save the image
    hdf5_file["test_img"][i, ...] = img[None]

    pass

# save the mean and close the hdf5 file
hdf5_file["train_mean"][...] = mean
hdf5_file.close()
