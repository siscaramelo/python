#!/usr/bin/env python
'''
Created on 08/01/2014

@author: pabgir
'''
from numpy import *

def computeCost(X, y, theta):
    
    # 
    # Compute the Cost Function for linear regression
    #
    
    # Intial values
    J = 0
    m = size(y)     # Number of training examples
    
    # The linear regression cost function
    J = dot(transpose(dot(X,theta)-y),dot(X,theta)-y)/(2*m)
    
    return J
    

def gradientDescent(X, y, theta, alpha, num_iters):    
    
    #
    #    Performs gradient descent to learn theta
    #
    
    m           = size(y)     # Number of training examples
    J_history   = zeros((num_iters,1))   # Inicializamos el vector de datos historicos
    
    for i in range(num_iters):
     
        # theta = theta - alpha/m*X'*(X*theta-y)
        theta = theta - alpha*(dot(transpose(X),dot(X,theta)-y))/m

        J_history[i] = computeCost(X, y, theta)
        
    return theta, J_history


def featureNormalize(X):
    
    #
    #    Normalize the features in X
    #
    
    X_norm  = X
    mu      = zeros((1, size(X, 1)))
    sigma   = zeros((1, size(X, 1)))
    
    # Compute the mean of each feature
    mu      = sum(X,0)/size(X,0)
    X_norm  = X-mu
    
    # Compute the standard deviation
    sigma  = std(X,0,ddof=1) # N-1
    X_norm = X_norm*power(sigma,-1)
    
    return X_norm, mu, sigma
    
    
def normalEqn(X, y):
    
    #
    #    Computes the closed-form solution to linear regression
    #    
    
    theta = zeros((size(X,1), 1))
    
    theta = dot(linalg.pinv(dot(transpose(X),X)),dot(transpose(X),y))
    
    return theta
    
    