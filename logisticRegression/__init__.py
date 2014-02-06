#!/usr/bin/env python
'''
Created on 15/01/2014

@author: pabgir
'''
from numpy import *

def sigmoid(Z):
    #
    #    Computes the sigmoid of z
    #
    
    g = zeros(shape(Z))       # Initialize g
    g = 1./(1.+power(e,-Z))           # Compute g
    
    return g

def predict(theta, X):
    #
    #    Computes the prediction for X using a threshold at 0.5
    #    with the learned logistic regression parameter theta
    #
    
    p = sigmoid(dot(X,c_[theta])) >= 0.5
    
    return p

def costFunction(theta, X, y):
    #
    #    Compute the Cost Function and Gradient Descent for Logistic Regression
    #
    
    #    Initial values
    m = size(y,0)      #  Number of training examples
    
    J = 0
    grad = zeros(shape(theta))
    theta = c_[theta]       # nx1 array
    
    #   Compute J and grad
    h = dot(X, theta)
    J = (-1*dot(transpose(y),log(sigmoid(h))) - dot(1-transpose(y),log(1-sigmoid(h)),))/m
         
    #    grad = dot(transpose(X),sigmoid(h)-y)/m
    #    grad = ndarray.flatten(grad)
    
    return J.item(0)##, grad.A

def costFunctionReg(theta, X, y, lambda1):
    #
    #    Compute the Cost Function and Gradient Descent for Logistic Regression
    #
    
    #    Initial values
    m = size(y,0)      #  Number of training examples
    
    J = 0
    grad = zeros(shape(theta))
    theta = c_[theta]       # nx1 array
    
    #   Compute J and grad
    # h = dot(X,reshape(theta,(1,size(theta))).T)
    h = dot(X, theta)
    J = (-1*dot(transpose(y),log(sigmoid(h)))-dot(1-transpose(y),log(1-sigmoid(h)),))/m + lambda1*dot(transpose(theta),theta)/(2*m)
                 
    grad = dot(transpose(X),sigmoid(h)-y)/m + lambda1*theta/m
    grad = ndarray.flatten(grad)
    
    return J.item(0), grad