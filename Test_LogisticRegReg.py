'''
Created on 22/01/2014

@author: pgiraldez
'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from logisticRegression import *
from numpy import newaxis, r_, c_, mat, e
from scipy import optimize


def XXXmapFeature(X1, X2):
    # MAPFEATURE Feature mapping function to polynomial features
    #
    # MAPFEATURE(X1, X2) maps the two input features
    # to quadratic features used in the regularization exercise.
    #
    # Returns a new feature array with more features, comprising of
    # X1, X2, X1**2, X2**2, X1*X2, X1*X2**2, etc..
    #
    # Inputs X1, X2 must be the same size
    #
    X1 = np.asarray(X1)
    X2 = np.asarray(X2)

    degree = 6
    out = [ones(size(X1))]
    for i in range(1, degree+1):
        for j in range(i+1):
            out.append(X1 ** (i-j) * X2 ** j)

    if isscalar(X1):
        return hstack(out) # if inputs are scalars, return a vector
    else:
        return column_stack(out)
    
def mapFeature(x1, x2):
    '''
    Maps the two input features to quadratic features.
 
    Returns a new feature array with more features, comprising of
    X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, etc...
 
    Inputs X1, X2 must be the same size
    '''
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    x1.shape = (x1.size, 1)
    x2.shape = (x2.size, 1)
    degree = 6
    out = ones(shape=(x1[:, 0].size, 1))
 
    m, n = out.shape
 
    for i in range(1, degree + 1):
        for j in range(i + 1):
            r = (x1**(i-j)) * (x2**j)
            out = append(out, r, axis=1)
 
    return out    
    
def plotData(X, y):
    # pos = (y.ravel() == 1).nonzero()
    # neg = (y.ravel() == 0).nonzero()
    pos = (y == 1).nonzero()[:1]
    neg = (y == 0).nonzero()[:1]
 
    plt.plot(X[pos, 0].T, X[pos, 1].T, 'k+', markeredgewidth=2, markersize=7)
    plt.plot(X[neg, 0].T, X[neg, 1].T, 'ko', markerfacecolor='r', markersize=7)

def plotDecisionBoundary(theta, X, y):
    #PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
    #the decision boundary defined by theta
    # PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the
    # positive examples and o for the negative examples. X is assumed to be
    # a either
    # 1) Mx3 matrix, where the first column is an all-ones column for the
    # intercept.
    # 2) MxN, N>3 matrix, where the first column is all-ones

    # Plot Data
    fig = plotData(X[:,1:], y)
    plt.hold(True)

    if size(X, 1) <= 2:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = array([min(X[:,1])-2, max(X[:,1])+2])

        # Calculate the decision boundary line
        plot_y = (-1/theta[2]) * (theta[1] * plot_x + theta[0])

        # Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y)

        # Legend, specific for the exercise
        plt.legend(('Admitted', 'Not admitted', 'Decision Boundary'), numpoints=1)
        plt.axis([30, 100, 30, 100])
    else:
        u = linspace(-1, 1.5, 50)
        v = linspace(-1, 1.5, 50)

        # Evaluate z = theta*x over the grid
        # z = frompyfunc(lambda x,y: mapFeature(x,y).dot(theta), 2, 1).outer(u,u)
        z = zeros((size(u), size(v)))
        for i in range(1,size(u)):
            for j in range(1,size(v)):
                z[i,j] = dot(mapFeature(u[i], v[j]),theta)

        z = z.T # important to transpose z before calling contour

        # Plot z = 0
        # Notice you need to specify the level as [0]
        plt.contour(u, v, z, [0], linewidth=2)

    plt.hold(False)

    return fig


if __name__ == '__main__':

    data = np.loadtxt('C:\Users\pgiraldez\Documents\Octave\mlclass-ex2\ex2data2.txt', delimiter=',')
    X = mat(c_[data[:, :2]])
    y = c_[data[:, 2]]

    # ============= Part 1: Plotting
     
    print 'Plotting data with + indicating (y = 1) examples and o ' \
    'indicating (y = 0) examples.'
     
    plotData(X, y)
    plt.ylabel('Microchip Test 1')
    plt.xlabel('Microchip Test 2')
    plt.legend(['y = 1', 'y = 0'])
    plt.show()
     
    raw_input('Press any key to continue\n')

    ## =========== Part 1: Regularized Logistic Regression ============
    #  In this part, you are given a dataset with data points that are not
    #  linearly separable. However, you would still like to use logistic 
    #  regression to classify the data points. 
    #
    #  To do so, you introduce more features to use -- in particular, you add
    #  polynomial features to our data matrix (similar to polynomial
    #  regression).
    #

    # Add Polynomial Features

    # Note that mapFeature also adds a column of ones for us, so the intercept
    # term is handled
    X = mapFeature(X[:,0], X[:,1])
     
    # Initialize fitting parameters
    initial_theta = np.zeros((size(X,1),1))

    # Set regularization parameter lambda to 1
    lambda1 = 0.07
    

    # Compute and display initial cost and gradient for regularized logistic
    # regression
    cost, grad = costFunctionReg(initial_theta, X, y, lambda1)#, None

    print 'Cost at initial theta (zeros): %f' % cost
    print 'Gradient at initial theta (zeros):\n%s' % grad
     
    raw_input('Press any key to continue\n')
     
   # ============= Part 3: Optimizing using fminunc
     
#     options = {'full_output': True, 'maxiter': 400}
#      
#     theta, cost, _, _, _ = \
#     optimize.fmin(lambda t: costFunctionReg(t, X, y, lambda1), initial_theta, **options)

    
#     res = optimize.minimize(costFunctionReg, initial_theta, args=(X,y,lambda1), \
#                         method='Nelder-Mead', options={'maxiter':400})
     
    res = optimize.minimize(costFunctionReg, initial_theta, args=(X,y,lambda1), \
                        method='BFGS', jac=True, options={'maxiter':400})
    theta = res.x
    cost = res.fun
   
    print 'Cost at theta found by fminunc: %f' % cost
    print 'theta: %s' % theta
     
    plotDecisionBoundary(theta, X, y)
    plt.ylabel('Microchip Test 1')
    plt.xlabel('Microchip Test 2')
    plt.legend(['y = 1', 'y = 0'])
    plt.title('Lambda = %f' % lambda1)
    plt.show()
     
    raw_input('Press any key to continue\n')
     
    # ============== Part 4: Predict and Accuracies
     
    # Compute accuracy on our training set
    p = predict(theta, X);

    print 'Train Accuracy: %f\n' % (mean(p == y) * 100)
    print '\nProgram paused. Press enter to continue.'

    raw_input('Press any key to continue\n')
     
