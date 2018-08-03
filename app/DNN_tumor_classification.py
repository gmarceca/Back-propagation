"""NN for classifying a tumor as benign or malign"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from utils.setup_tumor_classification import *
from utils.dnn_clasificacion import *

##%matplotlib inline
#get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#%load_ext autoreload
#%autoreload 2

classes = {0:'B', 1:'M'}
features = {0:'Radio',1:'Textura',2:'Perimetro',3:'Area',4:'Suavidad',5:'Compacidad',6:'Concavidad',7:'Puntos concavos',8:'Simetria',9:'Densidad'}


# Load dataset
train_x, train_y, val_x, val_y, test_x, test_y = load_data_RN()

print("training shape:", train_x.shape)
print("validating shape:", val_x.shape)
print("testing shape:", test_x.shape)

n_x = train_x.shape[0] #x.shape = 10*409
n_h = 3
#n_h2 = 10
n_y = 1
layers_dims = (n_x, n_h, n_y)

parameters = L_layer_model(train_x, train_y, val_x, val_y, layers_dims, learning_rate=1.0, num_iterations = 3000, sigma_init=0.2,print_cost=True)
