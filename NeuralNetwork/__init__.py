#!/usr/bin/env python
'''
Created on 27/01/2014

@author: pabgir
'''
from numpy import *
from logisticRegression import sigmoid
from numpy import newaxis, r_, c_, mat, e

def predict(Theta1, Theta2, X):
    #
    #    Output the label for an input X given a trained neural network Theta1, Theta2
    #

    # Initial values
    m = size(X, 0)
    num_labels = size(Theta2, 0)

    # You need to return the following variables correctly 
    p = zeros([m, 1])

    # Feedfoward propagation
    A1 = c_[ones(m), X]
    A2 = sigmoid(dot(A1,Theta1.T))   # Hidden layer

    A2    = c_[ones(m), A2]
#    p = sigmoid(dot(A2,Theta2.T)) # Output layer
    p = argmax(sigmoid(dot(A2,Theta2.T)),1) # Output layer
    
    return p

def sigmoidGradient(z):
    #
    #    Returns the gradient of the sigmoid function evaluated at z
    #
    
    g = zeros(shape(z))
    g = sigmoid(z)*(1.-sigmoid(z))
    
    return g
    
def nnH(Theta1,Theta2,X):
    #
    #    Compute the h(X) function - The NN result
    #
    
    #    Initial values
    m          = size(mat(X),0)
    num_labels = size(Theta2,0)
    h          = zeros([m, 1])
    
    A1 = c_[ones([m,1]),mat(X)]
    A2 = sigmoid(A1*Theta1.T)

    A2 = c_[ones([m,1]),A2]
    h = sigmoid(A2*Theta2.T)
    
    return h
    
def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, plambda):
    #
    #    Compute Cost Funtion for neural networks
    # 
    
    # Reshape nn_params back into the parameters Theta1 and Theta2
    Theta1 = reshape(nn_params[0:hidden_layer_size * (input_layer_size + 1)],[hidden_layer_size,input_layer_size+1])
    Theta2 = reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],[num_labels, hidden_layer_size + 1])
    
    # Initialize variables
    m = size(X,0)
    J = 0.
    Theta1_grad = zeros(shape(Theta1))
    Theta2_grad = zeros(shape(Theta2))
    
    # Turn y in a ones' matrix
    Y = zeros([m,num_labels])
    for i in range(m):
        Y[i,y[i]-1] = 1
        
    # First we compute the Cost Function 
    Y = Y.flatten()
    h = nnH(Theta1,Theta2,X).flatten()
    J = (-1.*Y.dot(log(h).T)-(1.-Y).dot(log(1.-h).T))/m
    
    # Regularize Cost Function
    J = J + plambda*(sum(Theta1[:,1:]**2)+sum(Theta2[:,1:]**2))/(2.*m)
    
    # Implement the Backpropagation algorithm
    # Compute delta
#     delta3 = h - Y
#     delta2 = delta3.dot(Theta2[:,1:])*sigmoidGradient(X.dot(Theta1.T))
    
    
    return J #, grad