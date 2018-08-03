import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

######################################## Other Functions ##############################################

def plotVars (x1_array, x2_array, y_array, row1, row2, col, features):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Select data slices
    #y_sub = [y for (ind,y) in enumerate(y_array) if (x_array.tolist()[0][ind]<0) and (-0.5 < x_array.tolist()[0][ind])]
    #x_sub = [x for (ind,x) in enumerate(x_array) if (x_array.tolist()[0][ind]<0) and (-0.5 < x_array.tolist()[0][ind])]

  #  print("X-array: ",x_array)
   # print("Y-array: ",y_array)

    ax.scatter(x1_array, x2_array, y_array, c='green', marker='o')

    ax.set_xlabel(features[row1])
    ax.set_ylabel(features[row2])
    ax.set_zlabel(features[col])

    #plt.plot(x1_array, x2_array, y_array,'o',color='green')
    #plt.xlabel(features[row1])
    #plt.ylabel(features[row2])
    #plt.zlabel(features[col])
    #plt.title("Regresion")
    #plt.legend()
    #plt.savefig('plots/test_{}_{}_vs_{}.pdf'.format(features[row1],features[row2],features[col]))

    plt.show()
    #plt.close()

    pass


def generate_data (N):

    x = np.random.uniform(0,1,N)
    y = np.sin(2*np.pi*x)

    print("Generated data with mu= ",np.mean(x))
    print("and std= ",np.std(x))

    y = y + np.random.normal(0,0.1,len(y))

    plt.plot(x, y, 'o')
    plt.savefig("test_with_noise.pdf")
    plt.close()

    return x, y


def fill_X_matrix (x, M):

    func = lambda x, j: x**(j+1)
    return np.array([[func(val,j) for j in range(M+1)] for val in x])

######################################## L-Layer NN model #############################################




def sigmoid(Z):

    A = 1/(1+np.exp(-Z))
    cache = Z

    return A, cache

def sigmoid_backward(dA, cache):
    Z = cache
    
    s = 1/(1+np.exp(-Z))

    dZ = dA * s * (1-s)

    assert(Z.shape == dZ.shape)

    return dZ


def linear_backward(dA, cache):
    Z = cache
    
    s = 1/(1+np.exp(-Z))

    dZ = dA * s * (1-s)

    assert(Z.shape == dZ.shape)

    return dZ


def relu(Z):

    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)

    cache = Z

    return A, cache

def relu_backward(dA, cache):

    Z = cache

    assert (dA.shape == Z.shape)

    return dA*((Z>0)*1)


def initialize_parameters_deep(layer_dims,sigma_init):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    #SIGMOID -> SIGMOID converges with np.random.seed(4)

    np.random.seed(4)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])* sigma_init
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1)) 
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters



def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    Z = np.dot(W, A) + b
 
    #print("Z1: ",Z)
    #print("W1: ",W)
    #print("b1: ",b)
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    elif activation == "linear":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = Z, Z
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache



def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    # Compute regression loss from aL and y.
    cost = (1 / (2*m)) * np.sum((Y - AL)**2)
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1. / m) * np.dot(dZ, cache[0].T) 
    db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(cache[1].T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
   
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)

    elif activation == "linear":
        dZ = dA


    #print("dZ: ",dZ)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->SIGMOID] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = (AL-Y)
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation="linear")
    
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        current_cache = caches[l]
        
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation="sigmoid")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->SIGMOID]*(L-1)->LINEAR->LINEAR computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, 
                                             parameters["W" + str(l)], 
                                             parameters["b" + str(l)], 
                                             activation='sigmoid')
        caches.append(cache)

    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, 
                                             parameters["W" + str(L)], 
                                             parameters["b" + str(L)], 
                                             activation='linear')
    caches.append(cache)
    
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches



def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural network


    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
        
    return parameters


def save (cost, cost_test):

    name = "calefaccion_cost_cfg4"

    cost_arr = np.asarray(cost)
    cost_test_arr = np.asarray(cost_test)

    x = np.arange(cost_arr.shape[0])

    print(type(cost_arr))
    print(cost_arr.shape)

    with open('utils/inputs_plotter/x_axis_{}.pkl.gz'.format(name), "wb") as output:
        pickle.dump(x, output)

    with open('utils/inputs_plotter/train_{}.pkl.gz'.format(name), "wb") as output:
        pickle.dump(cost_arr, output)

    with open('utils/inputs_plotter/test_{}.pkl.gz'.format(name), "wb") as output:
        pickle.dump(cost_test_arr, output)


def L_layer_model(X, Y, X_test, Y_test, layers_dims, category, sigma_init=0.5, mu=[0], std=[1], learning_rate = 0.0075, num_iterations = 3000, validation = False, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->SIGMOID]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mu -- mean of X data generated
    std -- sigma of X data generated
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps


    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    costs = []                         # keep track of cost
    costs_test = []                         # keep track of cost
    
    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims, sigma_init)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> SIGMOID]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        AL_test, caches_test = L_model_forward(X_test, parameters)
        
        # Compute cost.
        cost = compute_cost(AL, Y)
        cost_test = compute_cost(AL_test, Y_test)
    
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)
 
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate=learning_rate)
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            costs_test.append(cost_test)

    name = 'energy'

    if (validation):
        name = 'val'

    save(costs,costs_test)
    # plot the cost

    plt.plot(np.squeeze(costs),label='Train data')
    plt.plot(np.squeeze(costs_test),label='Test data')
    plt.legend()
    plt.ylabel('cost')
    plt.yscale("log",nonposy='clip')
    plt.xlabel('iterations (per cents)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.savefig("cost_reg_{}_{}.pdf".format(name,category))

    plt.close()

    # prediction

    if (validation):

        x_norm = (X[0,:].reshape(1,X.shape[1]))
        x_list = (X[0,:].reshape(1,X.shape[1])).tolist()
        y_list = Y.tolist()
        pred_list = AL.tolist()

        x_test_list = (X_test[0,:].reshape(1,X_test.shape[1])).tolist()
        y_test_list = Y_test.tolist()
        pred_test_list = AL_test.tolist()

        y_truth = (np.sin(2*np.pi*(x_norm*std[0]+mu[0])))

        plt.plot(x_list, y_list,'o',color='green',label='Generated data')
        plt.plot(x_list, pred_list,'^',color='blue',label='Predicted data')
        plt.plot(x_list, y_truth,'+', color='red',label='Truth data')

        plt.ylabel('Y')
        plt.xlabel('X')
        plt.title("Learning rate =" + str(learning_rate))

        handles, labels = plt.gca().get_legend_handles_labels()
        handle_list, label_list = [], []
        for handle, label in zip(handles, labels):
            if label not in label_list:
                handle_list.append(handle)
                label_list.append(label)
        plt.legend(handle_list, label_list)


        plt.savefig("data_pred_and_gen.pdf")

        plt.close()

    
    return parameters
