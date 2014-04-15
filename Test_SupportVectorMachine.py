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

from sklearn import svm, grid_search

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
    xp = linspace(min(X[:,0]), max(X[:,0]), 100)
    yp = - (w[0]*xp + b)/w[1]
    plotData(X, y,False)
    plt.plot(xp, yp, '-b') 
    plt.show()    
    
    
def visualizeBoundary(X, y, model, *args):
    # VISUALIZEBOUNDARY plots a non-linear decision boundary learned by the SVM
    #   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a non-linear decision 
    #   boundary learned by the SVM and overlays the data on it

    # Plot the training data on top of the boundary
    plotData(X, y, False)

    # Make classification predictions over a grid of values
    x1plot = linspace(min(X[:,0]), max(X[:,0]), 100)
    x2plot = linspace(min(X[:,1]), max(X[:,1]), 100)
    X1, X2 = meshgrid(x1plot, x2plot)
    vals = zeros(shape(X1))
    
    # Plot the SVM boundary
    vals = model.decision_function(c_[X1.ravel(),X2.ravel()])
    vals[vals>=1] = 1
    vals[vals<1]  = 0
    vals = vals.reshape(X1.shape)

#    plt.contour(X1, X2, vals, [0, 0], 'Color', 'b')
    plt.contour(X1, X2, vals, colors='b')
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
    y     = int_(data['y'].flatten()) 

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
#     model = svmTrain(X, y, C, linearKernel, 1e-3, 20)
    
    # Segunda via
    svmLinear = svm.SVC(kernel='linear', C=100)
    model.w = (svmLinear.fit(X, y).coef_).flatten()
    model.b = svmLinear.fit(X,y).intercept_
    
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


    ## ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
    #  After you have implemented the kernel, we can now use it to train the 
    #  SVM classifier.
    # 
    print('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n')

    # Load from ex6data2: 
    # You will have X, y in your environment
    data = loadmat('C:\Users\pgiraldez\Documents\Octave\mlclass-ex6\ex6data2.mat') 
    X     = data['X']
    y     = int_(data['y']).flatten() 

    # SVM Parameters
    C = 1
    sigma = 0.01

    # We set the tolerance and max_passes lower here so that the code will run
    # faster. However, in practice, you will want to run the training to
    # convergence.

    # Segunda via
    # Map 0 to -1
    Y = copy(y)
    Y[Y==0] = -1
    Y = Y.flatten()
    model = svm.SVC(kernel='rbf', C=100, gamma = 1/sigma)     # gamma = 1/ sigma
    model.fit(X,Y)
    
    visualizeBoundary(X, y, model)

    raw_input('Program paused 5. Press any key to continue\n')

    ## =============== Part 6: Visualizing Dataset 3 ================
    #  The following code will load the next dataset into your environment and 
    #  plot the data. 
    #

    print('Loading and Visualizing Data ...\n')

    # Load from ex6data3: 
    # You will have X, y in your environment
    data = loadmat('C:\Users\pgiraldez\Documents\Octave\mlclass-ex6\ex6data3.mat') 
    X     = data['X']
    y     = int_(data['y']).flatten() 

    # Plot training data
    plotData(X, y)

    raw_input('Program paused 6. Press any key to continue\n')


    ## ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========

    #  This is a different dataset that you can use to experiment with. Try
    #  different values of C and sigma here.
    # 

    # Load from ex6data3: 
    # You will have X, y in your environment
    data = loadmat('C:\Users\pgiraldez\Documents\Octave\mlclass-ex6\ex6data3.mat') 
    X     = data['X']
    y     = int_(data['y']).flatten() 

    # Try different SVM Parameters here
    parameters = {'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30], 'gamma': [300, 100, 30, 10, 3, 1]}
        
    #[C, sigma] = dataset3Params(X, y, Xval, yval);

    # Train the SVM
    svr=svm.SVC(kernel='rbf')
    model = grid_search.GridSearchCV(svr,parameters)
    model.fit(X,y)

    visualizeBoundary(X, y, model)

    raw_input('Program paused 7. Press any key to continue\n')
