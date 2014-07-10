'''
Created on 22/05/2014

@author: pgiraldez
'''


# Load Python libraries
from numpy import *
from numpy import newaxis, r_, c_, mat, e
from scipy import optimize
from scipy.io import loadmat
# from matplotlib import pyplot as plt
# import matplotlib.cm as cm

# from sklearn import svm, grid_search

from Clustering import *

if __name__ == '__main__':
    
    ## ================= Part 1: Find Closest Centroids ====================
    #  To help you implement K-Means, we have divided the learning algorithm 
    #  into two functions -- findClosestCentroids and computeCentroids. In this
    #  part, you shoudl complete the code in the findClosestCentroids function. 
    #
    print('Finding closest centroids.\n\n')

    # Load an example dataset that we will be using
    data = loadmat('C:\Users\pgiraldez\Documents\Octave\mlclass-ex7\ex7data2.mat') 
    X     = data['X'] 

    # Select an initial set of centroids
    K = 3       # 3 Centroids
    initial_centroids = array([[3, 3],[6, 2], [8, 5]])

    # Find the closest centroids for the examples using the
    # initial_centroids
    idx = findClosestCentroids(X, initial_centroids)

    print('Closest centroids for the first 3 examples: \n')
    print idx[0:3].flatten()
    print('\n(the closest centroids should be 1, 3, 2 respectively)\n')

    raw_input('Program paused 1. Press any key to continue\n')
 
    ## ===================== Part 2: Compute Means =========================
    #  After implementing the closest centroids function, you should now
    #  complete the computeCentroids function.
    #
    print('\nComputing centroids means.\n\n')

    #  Compute means based on the closest centroids found in the previous part.
    centroids = computeCentroids(X, idx, K)

    print('Centroids computed after initial finding of closest centroids: \n')
    print(centroids)
    print('\n(the centroids should be\n')
    print('   [ 2.428301 3.157924 ]')
    print('   [ 5.813503 2.633656 ]')
    print('   [ 7.119387 3.616684 ]\n')

    raw_input('Program paused 2. Press any key to continue\n')
