'''
Created on 08/01/2014

@author: pgiraldez
'''
import numpy as np
from linearRegression import *

if __name__ == '__main__':
    
    # Feature Normalize funtion check
    X = np.array([[1.,2.],[3.,4.],[5.,6.]])
    y = np.array([[10.],[16.],[22.]])
    
    print 'X = ',X
    print 'y = ',y
    
    X_norm, mu, sigma = featureNormalize(X)
    
    print
    print "--- Matriz normalizada ---"
    print X_norm, mu, sigma
    
    print
    print "--- Norma Equation ---"
    
    theta = normalEqn(X, y)
    print theta