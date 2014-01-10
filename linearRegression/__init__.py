#!/usr/bin/env python
'''
Created on 08/01/2014

@author: pabgir
'''
import numpy as np

def computeCost(X, y, theta):
    
    # 
    # Compute the Cost Function for linear regression
    #
    
    # Intial values
    J = 0
    m = np.size(y)     # Number of training examples
    
    # The linear regression cost function
    J = 1/(2*m)*(np.dot(np.transpose(np.dot(X,theta)-y),np.dot(X,theta)-y))
    
    return J
    

def gradientDescent(X, y, theta, alpha, num_iters):    
    
    #
    #    Performs gradient descent to learn theta
    #
    
    m           = np.size(y)     # Number of training examples
    J_history   = np.zeros((num_iters,1))   # Inicializamos el vector de datos historicos
    
    for i in range(num_iters):
     
        # theta = theta - alpha/m*X'*(X*theta-y)
        theta = theta - alpha*(np.dot(np.transpose(X),np.dot(X,theta)-y))/m

        J_history[i] = computeCost(X, y, theta)
        
    return theta, J_history


def featureNormalize(X):
    
    #
    #    Normalize the features in X
    #
    
    X_norm  = X
    mu      = np.zeros((1, np.size(X, 1)))
    sigma   = np.zeros((1, np.size(X, 1)))
    
    # Compute the mean of each feature
    mu      = np.sum(X,0)/np.size(X,1)
    X_norm  = X-mu
    
    # Compute the standard deviation
    sigma  = np.std(X)
    X_norm = np.array(X_norm)*np.array(np.power(sigma,-1))
    
    return X_norm, mu, sigma
    