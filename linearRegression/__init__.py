#!/usr/bin/env python
'''
Created on 08/01/2014

@author: pabgir
'''
from numpy import *
from scipy import optimize

def computeCost(X, y, theta, plambda=0):
    
    # 
    # Compute the Cost Function for linear regression
    #
    
    # Intial values
    J = 0.
    m = size(y)     # Number of training examples
    
    # The linear regression cost function
    J = dot(transpose(dot(X,theta)-y),dot(X,theta)-y)/(2.*m) + plambda*dot(transpose(theta),theta)/(2.*m)
    
    return J
    

def gradientDescent(X, y, theta, alpha, num_iters, plambda = 0):    
    
    #
    #    Performs gradient descent to learn theta
    #
    
    m           = size(y)     # Number of training examples
    J_history   = zeros((num_iters,1))   # Inicializamos el vector de datos historicos
    
    for i in range(num_iters):
     
        # theta = theta - alpha/m*X'*(X*theta-y)
        theta = theta - alpha*(dot(transpose(X),dot(X,theta)-y))/m + plambda*theta/m

        J_history[i] = computeCost(X, y, theta)
        
    return theta, J_history

def linearRegCostFunction(X, y, theta, plambda=0):
    # 
    # Compute the Cost Function for linear regression
    #
    
    # Intial values
    J = 0.
    m = size(y)     # Number of training examples
    thetaReg = copy(theta)
    thetaReg[0] = 0
    
    # The linear regression cost function
    try:
        J = dot(transpose(dot(X,theta)-y),dot(X,theta)-y)/(2.*m) + plambda*dot(transpose(thetaReg),thetaReg)/(2.*m)
    except:
        print shape(X),shape(theta), shape(thetaReg), shape(y)
        raise

    # Afterwards compute the Gradient Descent
    # grad = X'*(X*theta-y)/m;
    grad =  (dot(transpose(X),dot(X,theta)-y))/m + plambda*thetaReg/m
    
    return J, grad

def trainLinearReg(X, y, plambda=0):
    # Initialize Theta
    n = X.shape[1]
    #initial_theta = zeros([size(X, 1), 1])
    initial_theta = zeros(n)

    # Create "short hand" for the cost function to be minimized
    costFunction = lambda p: linearRegCostFunction(X, y, p, plambda)

    # Now, costFunction is a function that takes in only one argument

    # Minimize using fmincg # Ver 'GradObj': 'on',
    res = optimize.minimize(costFunction, initial_theta, method='CG', jac=True, options={'maxiter': 200, 'disp':False})
    
    theta = res.x
    cost = res.fun
    
    return theta
    
def learningCurve(X, y, Xval, yval, plambda):
    #
    #    Generates the train and cross validation set errors needed to plot a learning curve
    #
    
    # Number of training examples
    m = size(X, 0)

    # You need to return these values correctly
    error_train = zeros(m+1)
    error_val   = zeros(m+1)
    
    for i in range(1,m+1):
            
        theta = trainLinearReg(X[:i,:], y[:i], plambda)
        error_train[i], grad = linearRegCostFunction(X[:i,:], y[:i], theta, 0.)
        error_val[i], grad   = linearRegCostFunction(Xval, yval, theta, 0.)
    
    return error_train, error_val

def validationCurve(X, y, Xval, yval):
    #
    #    Returns the train  and validation errors (in error_train, error_val)
    #       for different values of lambda
    #

    # Selected values of lambda (you should not change this)
    lambda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    
    error_train = zeros([size(lambda_vec), 1])
    error_val = zeros([size(lambda_vec), 1])

    for i in range(1,size(lambda_vec)):
        plambda = lambda_vec[i]
        # Compute train / val errors when training linear 
        # regression with regularization parameter lambda
        # You should store the result in error_train(i)
        # and error_val(i)

        theta = trainLinearReg(X, y, plambda)
        error_train[i],grad = linearRegCostFunction(X, y, theta, 0)
        error_val[i],grad   = linearRegCostFunction(Xval, yval, theta, 0)
    
    return lambda_vec, error_train, error_val


def featureNormalize(X, mu=None, sigma=None):
    
    #
    #    Normalize the features in X
    #
    
    X_norm  = X
    
    # Compute the mean of each feature
    if mu is None:
        mu      = zeros((1, size(X, 1)))
        mu      = sum(X,0)/size(X,0)
        
    X_norm  = X-mu
    
    # Compute the standard deviation
    if sigma is None:
        sigma   = zeros((1, size(X, 1)))
        sigma  = std(X,0,ddof=1) # N-1
        
    X_norm = X_norm * power(sigma,-1)
    
    return X_norm, mu, sigma
    
def polyFeatures(X, p):
    #
    #   Takes a data matrix X (size m x 1) and maps each example into its polynomial features where
    #   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
    #

    X_poly = copy(X) 
         
    # Given a vector X, return a matrix X_poly where the p-th column of X contains the values of X to the p-th power.
#    for i in range(2,p):
    for i in range(2,p+1):
        X_poly = c_[X_poly, X**i]
     
    return X_poly

def polyFeaturesXXX(X, p):
     """
         Creates polynomial features up to ``power``.
     """
     cols = [power(asarray(X),p) for p in range(p + 1)]
     return vstack(cols).T    
    
def normalEqn(X, y):
    
    #
    #    Computes the closed-form solution to linear regression
    #    
    
    theta = zeros((size(X,1), 1))
    
    theta = dot(linalg.pinv(dot(transpose(X),X)),dot(transpose(X),y))
    
    return theta
    
    