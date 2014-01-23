'''
Created on 08/01/2014

@author: pgiraldez
'''
from numpy import *
import matplotlib.pyplot as plt
from linearRegression import *
import csv

if __name__ == '__main__':
    
    # Feature Normalize funtion check
    X = array([[1.,2.],[3.,4.],[5.,6.]])
    y = array([[10.],[16.],[22.]])

    #    Reading data from a file
    F = genfromtxt('C:\Users\pgiraldez\Documents\Octave\mlclass-ex1\ex1data2.txt',delimiter=',')

    X = F[:,[0,1]]
    y = F[:,2]
    y = reshape(y,(47,1))
    
    print 'X = ',X
    print 'y = ',y
    
    X_norm, mu, sigma = featureNormalize(X)
    
    print
    print "--- Matriz normalizada ---"
    print X_norm, mu, sigma
    
    print
    print "--- Normal Equation ---"

    X0 = ones([size(X,0),1])
    X=concatenate([X0,X_norm],1)
    
    theta = normalEqn(X, y)
    print theta
    
    print
    print '--- Running Gradient Descent ---'
    
    alpha = 0.1
    num_iters = 400
    theta = zeros([size(X,1), 1])
    theta, J_history = gradientDescent(X, y, theta, alpha, num_iters)
    
    print theta
    # print
    # print J_history
    
    plt.plot(range(1,num_iters+1),J_history)
    plt.title('Cost function evolution')
    plt.xlabel('Iteration number')
    plt.ylabel('Cost')
    plt.show()

    
    #print dim(F), dim(X), dim(y)