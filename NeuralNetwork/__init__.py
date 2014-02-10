#!/usr/bin/env python
'''
Created on 27/01/2014

@author: pabgir
'''
from numpy import *
from logisticRegression import sigmoid
from numpy import newaxis, r_, c_, mat, e
from numpy.linalg import norm

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

def randInitializeWeights(L_in, L_out):
    #
    #    Initialize arrays randomly to break the symmetry while training the NN
    #
    #    L_in: incoming connections
    #    L_out: outgoing connections
    
    W = zeros([L_out,1+L_in])
    
    # Randomly initialize the weights to small values
    epsilon_init = 0.12
    W = random.rand(L_out,1+L_in)*2*epsilon_init-epsilon_init
    
    return W
    
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
    h = nnH(Theta1,Theta2,X)
    Yf = Y.flatten()
    Hf = h.flatten()
    J = (-1.*Yf.dot(log(Hf).T)-(1.-Yf).dot(log(1.-Hf).T))/m
    
    # Regularize Cost Function
    J = J + plambda*(sum(Theta1[:,1:]**2)+sum(Theta2[:,1:]**2))/(2*m)
    
    # Implement the Backpropagation algorithm
    # Compute delta
    A1 = c_[ones([m,1]),X]
    A2 = c_[ones([m,1]),sigmoid(A1.dot(Theta1.T))]
    
    delta3 = h - Y
    delta2 = multiply(delta3.dot(Theta2[:,1:]),sigmoidGradient(A1.dot(Theta1.T)))
    
    Theta2bias = copy(Theta2); Theta2bias[:,0] = 0;
    Theta1bias = copy(Theta1); Theta1bias[:,0] = 0;

    Theta2_grad = delta3.T.dot(A2)/m + plambda*Theta2bias/m
    Theta1_grad = delta2.T.dot(A1)/m + plambda*Theta1bias/m
    
    grad = c_[Theta1_grad.flatten(),Theta2_grad.flatten()]
    
    return J, grad

def debugInitWeights(fan_out, fan_in):
    return sin(1 + arange((1 + fan_in) * fan_out)).reshape((1 + fan_in, fan_out)).T / 10.0

def checkNNGradients(plambda=0.0):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # We generate some 'random' test data
    Theta1 = debugInitWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitWeights(num_labels, hidden_layer_size)

    # Reusing debugInitializeWeights to generate X
    X = debugInitWeights(m, input_layer_size - 1)
    y = 1 + arange(1, m + 1) % num_labels

    # Unroll parameters
    nn_params = concatenate((Theta1.flatten(), Theta2.flatten()))
    cost_func = lambda p: nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, plambda)
    cost, grad = cost_func(nn_params)
    num_grad = computeNumericalGradient(cost_func, nn_params)
    print(vstack((grad, num_grad)).T)
    print('The above two columns you get should be very similar.')
    print('(Left-Your Numerical Gradient, Right-Analytical Gradient)')
    diff = norm(num_grad - grad) / norm(num_grad + grad)
    print('If your backpropagation implementation is correct, then')
    print('the relative difference will be small (less than 1e-9).')
    print('Relative Difference: %g' % diff)


def computeNumericalGradient(cost_func, theta):
    numgrad = zeros_like(theta)
    perturb = zeros_like(theta)
    eps = 1e-4
    for p in range(theta.size):
        perturb[p] = eps
        loss1, _ = cost_func(theta - perturb)
        loss2, _ = cost_func(theta + perturb)
        numgrad[p] = (loss2 - loss1) / (2.0 * eps)
        perturb[p] = 0.0
    return numgrad