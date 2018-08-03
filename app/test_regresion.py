"""NN for a regression example"""

import numpy as np
import h5py

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from utils.setup_energia_regresion import *
from utils.dnn_energia_regresion import *

#from testCases import *
#from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward

np.set_printoptions(threshold=np.inf)

import matplotlib
matplotlib.get_backend()

#get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#%load_ext autoreload
#%autoreload 2


train_x, train_y = generate_data(100)
test_x, test_y = generate_data(30)

tra_x = fill_X_matrix(train_x,6)
tes_x = fill_X_matrix(test_x,6)

# Normalize the data
mu = np.mean(tra_x,axis=0)
std = np.std(tra_x,axis=0)

print(mu)
print(std)

tr_x = ((tra_x - mu)/std).T
te_x = ((tes_x - mu)/std).T

tr_y = train_y.reshape(1,train_y.shape[0])
te_y = test_y.reshape(1,test_y.shape[0])

print(tr_x.shape)
print(te_x.shape)
print(tr_y.shape)

#plt.plot(train_x, train_y,'o',color='green')
#plt.show()

n_x = tr_x.shape[0]
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)

parameters = L_layer_model(tr_x, tr_y, te_x, te_y, layers_dims, mu=mu, std=std, learning_rate=0.3, num_iterations = 1500, validation=True, print_cost=True)
