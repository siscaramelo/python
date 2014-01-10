'''
Created on 08/01/2014

@author: pgiraldez
'''
import numpy as np
from linearRegression import *

if __name__ == '__main__':
    
    X = np.matrix([[1,2],[3,4],[5,6]])
    
    print X
    
    X_norm, mu, sigma = featureNormalize(X)
    
    print "--- Matriz normalizada ---"
    print X_norm