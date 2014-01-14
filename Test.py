'''
Created on 08/01/2014

@author: pgiraldez
'''
import numpy as np
import matplotlib.pyplot as plt
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
    print "--- Normal Equation ---"

    X0 = np.ones([np.size(X,0),1])
    X=np.concatenate([X0,X],1)
    
    theta = normalEqn(X, y)
    print theta
    
    print
    print '--- Running Gradient Descent ---'
    
    alpha = 0.01
    num_iters = 1400
    theta = np.zeros([3, 1])
    theta, J_history = gradientDescent(X, y, theta, alpha, num_iters)
    
    print theta
    #print J_history
    
    plt.plot(range(1,num_iters+1),J_history)
    plt.title('Cost function evolution')
    plt.xlabel('Iteration number')
    plt.ylabel('Cost')
    plt.show()
