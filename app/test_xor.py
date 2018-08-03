"""NN for testing the XOR gate"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from utils.setup import *
from utils.dnn_clasificacion import *

#from testCases import *
#from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward

np.set_printoptions(threshold=np.inf)

##%matplotlib inline
#get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#np.random.seed(1)

# Load dataset
train_x = np.array([[0,1,0,1],[0,0,1,1]])
train_y = np.array([[0,1,1,0]])

print("x: ",train_x)
print("y: ",train_y)

n_x = train_x.shape[0]
n_h = 2
n_y = 1
layers_dims = (n_x, n_h, n_y)

#parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), learning_rate=0.5, num_iterations = 3000, print_cost=True)
parameters = L_layer_model(train_x, train_y, train_x, train_y, layers_dims = (n_x, n_h, n_y), learning_rate=0.5, num_iterations = 3000, sigma_init=0.5, validation=True, print_cost=True)

