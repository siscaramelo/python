#!/usr/bin/env python
'''
Created on 08/01/2014

@author: pabgir
'''
from numpy import *
from numpy import newaxis, r_, c_, mat, e
from matplotlib import pyplot as plt
import matplotlib.cm as cm

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

def kMeansInitCentroids(X, K):
    # Initialize K centroids to be used in the dataset X
    centroids = zeros([K, size(X, 1)])
    
    # Randomly reorder the indices of examples
    randidx = random.permutation(size(X, 0))
    
    # Take the first K examples as centroids
    centroids = X[randidx[1:K], :]

    return centroids

def runkMeans(X, initial_centroids, max_iters, plot_progress=False):
    # Initialize values
    m,n                 = shape(X)
    K                   = size(initial_centroids, 0)
    centroids           = initial_centroids
    previous_centroids  = centroids
    idx                 = zeros([m, 1])
    
    # Run K-Means
    for i in range(max_iters):
    
        # Output progress
        print('K-Means iteration ...', i+1, max_iters)
    
        # For each example in X, assign it to the closest centroid
        idx = findClosestCentroids(X, centroids)
    
        # Optionally, plot progress here
        if plot_progress:
            plotProgresskMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
            print('Press enter to continue...\n')
            #pause;
    
            # Given the memberships, compute new centroids
            centroids = computeCentroids(X, idx, K)
    
    return centroids, idx


def plotDataPoints(X, idx, K):
    
    plt.scatter(X[:,0],X[:,1], s=15)
    #plt.show()
    
    return

def plotProgresskMeans(X, centroids, previous, idx, K, i):
    
    # Plot the examples
    plotDataPoints(X, idx, K);

    # Plot the centroids as red dots
    plt.plot(centroids[:,0], centroids[:,1], 'ro') #'r', marker='x', markersize=10,  linewidth=0) #

    # Plot the history of the centroids with lines
    for j in range(size(centroids,0)):
        plt.plot([centroids[j,0], previous[j,0]],[centroids[j,1], previous[j,1]],'g-')
        
    # Title
    plt.title('Iteration number: %d' % (i+1))
    plt.show()
    return