"""NN for predicting calafaction power for a given building"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from utils.setup_energia_regresion import *
from utils.dnn_energia_regresion import *

#from testCases import *
#from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward

##%matplotlib inline
#get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#%load_ext autoreload
#%autoreload 2

np.random.seed(1)

features = {0:'Compacidad Relativa',1:'Area de la Superficie Total',2:'Area de las Paredes',3:'Area del Techo',4:'Altura Total',5:'Orientacion',6:'Area de Reflejo Total',7:'Distribucion del Area de Reflejo'}


# Load dataset

train_x, train_y, val_x, val_y = load_data_RN()

#x.shape = 8*499
#y.shape = 2*499

sel = {'cal':0,'ref':1}
category = 'cal'

train_y_cal = train_y[sel[category],:].reshape(1, train_y.shape[1])
test_y_cal = val_y[sel[category],:].reshape(1, val_y.shape[1])


n_x = train_x.shape[0] 
n_h = 7
n_h2 = 7
n_h3 = 7
n_y = 1
layers_dims = (n_x, n_h, n_h2, n_h3, n_y)

parameters = L_layer_model(train_x, train_y_cal, val_x, test_y_cal, layers_dims, category, sigma_init=0.3, learning_rate=0.01, num_iterations = 60000, print_cost=True)
