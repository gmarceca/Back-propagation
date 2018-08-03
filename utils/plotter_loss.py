#!/usr/bin/env python

import numpy as np
rng = np.random.RandomState(21)  # For reproducibility
import pandas as pd

import matplotlib.pyplot as plt

import gzip
import pickle



def OpenFile (fi,opt):

    with open(fi, opt) as f:
        x = pickle.load(f)
        pass

    return x

# Main function definition
def main ():


    #inputs needed
    x = OpenFile('inputs_plotter/x_axis_calefaccion_cost_cfg1.pkl.gz', 'rb')
    
    ll_train_cfg1 = OpenFile('inputs_plotter/train_calefaccion_cost_cfg1.pkl.gz', 'rb')
    ll_test_cfg1 = OpenFile('inputs_plotter/test_calefaccion_cost_cfg1.pkl.gz', 'rb')
    
    ll_train_cfg2 = OpenFile('inputs_plotter/train_calefaccion_cost_cfg2.pkl.gz', 'rb')
    ll_test_cfg2 = OpenFile('inputs_plotter/test_calefaccion_cost_cfg2.pkl.gz', 'rb')
    ll_train_cfg3 = OpenFile('inputs_plotter/train_calefaccion_cost_cfg3.pkl.gz', 'rb')
    ll_test_cfg3 = OpenFile('inputs_plotter/test_calefaccion_cost_cfg3.pkl.gz', 'rb')
    ll_train_cfg4 = OpenFile('inputs_plotter/train_calefaccion_cost_cfg4.pkl.gz', 'rb')
    ll_test_cfg4 = OpenFile('inputs_plotter/test_calefaccion_cost_cfg4.pkl.gz', 'rb')
#    ll_train_cfg5 = OpenFile('inputs_plotter/train_cost_cfg5.pkl.gz', 'rb')
#    ll_test_cfg5 = OpenFile('inputs_plotter/test_cost_cfg5.pkl.gz', 'rb')



    means_train_cfg1 = np.array(ll_train_cfg1)
    means_test_cfg1 = np.array(ll_test_cfg1)
    means_train_cfg2 = np.array(ll_train_cfg2)
    means_test_cfg2 = np.array(ll_test_cfg2)
    means_train_cfg3 = np.array(ll_train_cfg3)
    means_test_cfg3 = np.array(ll_test_cfg3)
    means_train_cfg4 = np.array(ll_train_cfg4)
    means_test_cfg4 = np.array(ll_test_cfg4)
#    means_train_cfg5 = np.array(ll_train_cfg5)
#    means_test_cfg5 = np.array(ll_test_cfg5)




    fig, ax = plt.subplots()
    plt.plot(x, means_train_cfg4, 'c', label='Training,'+ r' $\lambda$=0.01, n_h1=7, n_h2=7, n_h3=7, h_L=3')
    plt.plot(x, means_test_cfg4, 'c--', label='Validation,'+ r' $\lambda$=0.01, n_h1=7, n_h2=7, n_h3=7, h_L=3')
    plt.plot(x, means_train_cfg3, 'm', label='Training,' + r' $\lambda$=0.01, n_h=5, h_L=1')
    plt.plot(x, means_test_cfg3, 'm--', label='Validation,' + r' $\lambda$=0.01, n_h=5, h_L=1')
#    plt.plot(x, means_train_cfg1, 'k', label='Training,'+ r' $\lambda$=0.01, n_h1=7, n_h2=5, h_L=2')
#    plt.plot(x, means_test_cfg1, 'k--', label='Validation,'+ r' $\lambda$=0.01, n_h1=7, n_h2=5, h_L=2')
#    plt.plot(x, means_train_cfg2, 'r', label='Training,'+ r' $\lambda$=0.1, n_h1=7, n_h2=5, h_L=2')
#    plt.plot(x, means_test_cfg2, 'r--', label='Validation,'+ r' $\lambda$=0.1, n_h1=7, n_h2=5, h_L=2')
    plt.yscale("log",nonposy='clip') 
    
    
    plt.legend()
    plt.xlabel("Boosting step / no. estimator")
    plt.ylabel("Log-loss")
    

    # Save
    plt.savefig('./logloss_test.pdf')


    return


# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    #args = parse_args()

    # Call main function
    main()
    pass

