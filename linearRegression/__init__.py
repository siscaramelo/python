#!/usr/bin/env python
'''
Created on 13/05/2013

@author: pabgir
'''
import numpy as np

def computeCost(X, y, theta):
    
    # 
    # Compute the Cost Function for linear regression
    #
    
    # Intial values
    computeCost = 0
    m           = np.size(y)     # Number of training examples
    
    # The linear regression cost function
    computeCost = 1/(2*m)*(np.dot(np.transpose(np.dot(X,theta)-y),(np.dot(X,theta)-y)))