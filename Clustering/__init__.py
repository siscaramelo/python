#!/usr/bin/env python
'''
Created on 08/01/2014

@author: pabgir
'''
from numpy import *
from numpy import newaxis, r_, c_, mat, e

def findClosestCentroids(X,centroids):
    # Returns the closest centroids for a dataset X where each row is a single example
    # idx = m * 1 vector of centroid assignments
    #
    
    # Set K
    K = size(centroids, 0)
    
    # Initialize centroid assignments vector
    idx = zeros([size(X,0),1])
    
    # idx(i) should contain the index of the centroid
    # closest to example i. Hence, it should be a value in the range 1..K
    distance_min = 0.0
    for i in range(size(X,0)):
        for j in range(K):
            distance = (X[i,:]-centroids[j,:]).dot((X[i,:]-centroids[j,:]).T)
            if j==0 or distance < distance_min:
                distance_min = distance
                idx[i] = j+1
    
    return idx

def computeCentroids(X,idx,K):
    # Return the new centroids by computing the means of the data points assigned to each centroid
    
    # Some variables
    m,n = shape(X)
    
    #Initialize centroids
    centroids = zeros([K,n])
    
    # Go over every centroid and compute mean of all points that belong to it.
    MIdx = []
    
    for k in range(K):
        col  = idx*(idx == k+1)/sum(idx[idx == k+1])
        if k==0 :
            MIdx = col
        else:
            MIdx = c_[MIdx, col]
        
    centroids = MIdx.T.dot(X)
    
    return centroids

