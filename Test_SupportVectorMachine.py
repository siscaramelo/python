'''
Created on 05/03/2014

@author: pgiraldez
'''

# Load Python libraries
from numpy import *
from numpy import newaxis, r_, c_, mat, e
from scipy import optimize
from scipy.io import loadmat
from matplotlib import pyplot as plt
import matplotlib.cm as cm

from SupportVectorMachine import *

# Funtions definition
def plotData(X, y, show=True):
    pos = (y == 1).nonzero()[:1]
    neg = (y == 0).nonzero()[:1]
 
    plt.plot(X[pos, 0].T, X[pos, 1].T, 'k+', markeredgewidth=2, markersize=7)
    plt.plot(X[neg, 0].T, X[neg, 1].T, 'ko', markerfacecolor='r', markersize=7)
    if show:
        plt.show()
    
def visualizeBoundaryLinear(X, y, model):
    #
    #   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a linear decision boundary 
    #   learned by the SVM and overlays the data on it
    #

    w = model.w
    b = model.b
    xp = linspace(min(X[:,1]), max(X[:,1]), 100)
    yp = - (w[0]*xp + b)/w[1]
    plotData(X, y,False)
    plt.plot(xp, yp, '-b') 
    plt.show()    
    
# ---------------------
#    Main program
# ---------------------
if __name__ == '__main__':
    
    ## =============== Part 1: Loading and Visualizing Data ================
    #  We start the exercise by first loading and visualizing the dataset. 
    #  The following code will load the dataset into your environment and plot
    #  the data.
    #

    print('Loading and Visualizing Data ...\n')
    
    # Load data from the example file
    data = loadmat('C:\Users\pgiraldez\Documents\Octave\mlclass-ex6\ex6data1.mat') 
    X     = data['X']
    y     = data['y'].flatten() 

    # Plot training data
    plotData(X, y)

    raw_input('Program paused 1. Press any key to continue\n')
    
    # ==================== Part 2: Training Linear SVM ====================
    #  The following code will train a linear SVM on the dataset and plot the
    #  decision boundary learned.
    #

    print('\nTraining Linear SVM ...\n')

    # You should try to change the C value below and see how the decision
    # boundary varies (e.g., try C = 1000)
    C = 1
    model = model()
    model = svmTrain(X, y, C, linearKernel, 1e-3, 20)
    visualizeBoundaryLinear(X, y, model)    

    raw_input('Program paused 2. Press any key to continue\n')
    
    ## =============== Part 3: Implementing Gaussian Kernel ===============
    #  You will now implement the Gaussian kernel to use
    #  with the SVM. You should complete the code in gaussianKernel.m
    #
    print('\nEvaluating the Gaussian Kernel ...\n')

    x1 = array([1, 2, 1])
    x2 = array([0, 4, -1])
    sigma = 2.
    
    sim = gaussianKernel(x1, x2, sigma)

    print('Gaussian Kernel between x1 = [1, 2, 1], x2 = [0, 4, -1], sigma = 0.5 : %f' % sim)
    print('\n(this value should be about 0.324652)\n')    

    raw_input('Program paused 3. Press any key to continue\n')

    ## =============== Part 4: Visualizing Dataset 2 ================
    #  The following code will load the next dataset into your environment and 
    #  plot the data. 
    #

    print('Loading and Visualizing Data ...\n')

    # Load from ex6data2: 
    # You will have X, y in your environment
    data = loadmat('C:\Users\pgiraldez\Documents\Octave\mlclass-ex6\ex6data2.mat') 
    X     = data['X']
    y     = data['y'].flatten() 

    # Plot training data
    plotData(X, y)

    raw_input('Program paused 4. Press any key to continue\n')
