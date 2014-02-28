'''
Created on 11/02/2014

@author: pgiraldez
'''

# Load Python libraries
from numpy import *
from linearRegression import *
from numpy import newaxis, r_, c_, mat, e
from matplotlib import pyplot as plt
from scipy import optimize
from scipy.io import loadmat
import matplotlib.cm as cm


def plot_fit(min_x, max_x, mu, sigma, theta, power):
    x           = arange(min_x , max_x , 0.05)
    Xpoly       = polyFeatures(x, power)
    Xpoly, _, _ = featureNormalize(Xpoly, mu, sigma)
    Xpoly = c_[ones([Xpoly.shape[0],1]),Xpoly]
    plt.plot(x, Xpoly.dot(theta), 'b', linewidth=2.0)

if __name__ == '__main__':
    
    ## =========== Part 1: Loading and Visualizing Data =============
    #  We start the exercise by first loading and visualizing the dataset. 
    #  The following code will load the dataset into your environment and plot
    #  the data.
    #

    # Load Training Data
    print('Loading and Visualizing Data ...\n')

    # Load from ex5data1: 
    # You will have X, y, Xval, yval, Xtest, ytest in your environment
    data = loadmat('C:\Users\pgiraldez\Documents\Octave\mlclass-ex5\ex5data1.mat') 
    X     = data['X']
    Xval  = data['Xval']
    Xtest = data['Xtest']
    y     = data['y'].flatten() 
    yval  = data['yval'].flatten()
    ytest = data['ytest'].flatten() 
     

    # m = Number of examples
    m = size(X, 0)
    
    # Plot training data
    plt.plot(X, y, 'ko', markerfacecolor='r', markersize=8)
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.show()
    
    raw_input('Program paused 1. Press any key to continue\n')
    
    ## =========== Part 2: Regularized Linear Regression Cost =============
    #  You should now implement the cost function for regularized linear 
    #  regression. 
    #

    theta = array([1 , 1])
    X1 = c_[ones([m, 1]), X]
    J, grad = linearRegCostFunction(X1, y, theta, 1.)

    print 'Cost at theta = [1 , 1]: %f'  %J      
    print '\n(this value should be about 303.993192)\n'

    raw_input('Program paused 2. Press any key to continue\n')

    ## =========== Part 3: Regularized Linear Regression Gradient =============
    #  You should now implement the gradient for regularized linear 
    #  regression.
    #

    print ('Gradient at theta = [1 ; 1]:  [%f, %f]' % (grad[0], grad[1])) 
    print '\n(this value should be about [-15.303016; 598.250744])\n'
         
    raw_input('Program paused 3. Press any key to continue\n')
    
    ## =========== Part 4: Train Linear Regression =============
    #  Once you have implemented the cost and gradient correctly, the
    #  trainLinearReg function will use your cost function to train 
    #  regularized linear regression.
    # 
    #  Write Up Note: The data is non-linear, so this will not give a great 
    #                 fit.
    #

    #  Train linear regression with lambda = 0
    plambda = 0
    theta = trainLinearReg(X1, y, plambda)
    
    #  Plot fit over the data
    h = dot(c_[ones([m, 1]), X],theta)
    print h
    plt.plot(X, y, 'ko', markerfacecolor='r', markersize=8)
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.plot(X, h,'b',linewidth=2.0)
    plt.show()

    raw_input('Program paused 4. Press any key to continue\n')

    ## =========== Part 5: Learning Curve for Linear Regression =============
    #  Next, you should implement the learningCurve function. 
    #
    #  Write Up Note: Since the model is underfitting the data, we expect to
    #                 see a graph with "high bias" -- slide 8 in ML-advice.pdf 
    #

    plambda = 0;
    error_train, error_val = learningCurve(c_[ones([m, 1]), X], y, c_[ones([size(Xval, 0), 1]), Xval], yval, plambda)
    
    plt.plot(arange(1,m+1),error_train[1:m+1],'r', arange(1,m+1), error_val[1:m+1],'b')
    plt.title('Learning curve for linear regression')
    plt.legend(('Train', 'Cross Validation'),numpoints=2)
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.axis([0, 13, 0, 150])
    plt.show()

    print('# Training Examples\tTrain Error\tCross Validation Error')
    for i in range(1,m+1):
        print('  \t%d\t\t%f\t%f' % (i, error_train[i], error_val[i]))

    raw_input('Program paused 5. Press any key to continue\n')

    ## =========== Part 6: Feature Mapping for Polynomial Regression =============
    #  One solution to this is to use polynomial regression. You should now
    #  complete polyFeatures to map each example into its powers

    power = 8
    Xpoly            = polyFeatures(X[:, 0], power)
    Xpoly, mu, sigma = featureNormalize(Xpoly)
    Xpoly = c_[ones(m),Xpoly]

    # Map Xtest_poly and normalize (using mu and sigma)
    Xtest_poly       = polyFeatures(Xtest[:, 0], power)
    Xtest_poly, _, _ = featureNormalize(Xtest_poly, mu, sigma)
    Xtest_poly = c_[ones(Xtest_poly.shape[0]),Xtest_poly]
    
    # Map Xval_poly and normalize (using mu and sigma)
    Xval_poly       = polyFeatures(Xval[:, 0], power)
    Xval_poly, _, _ = featureNormalize(Xval_poly, mu, sigma)
    Xval_poly = c_[ones(Xval_poly.shape[0]),Xval_poly]
    
    print('Normalized Training Example 1:\n%s' % Xpoly[0, :])
    
    raw_input('\nProgram paused 6. Press any key to continue\n')
    
    ## =========== Part 7: Learning Curve for Polynomial Regression =============
    #  Now, you will get to experiment with polynomial regression with multiple
    #  values of lambda. The code below runs polynomial regression with 
    #  lambda = 0. You should try running the code with different values of
    #  lambda to see how the fit and learning curve change.
    #
    
    plambda = 0
    theta = trainLinearReg(Xpoly, y, plambda)
    
    # Plot training data and fit
    plt.plot(X, y, 'ko', markerfacecolor='r', markersize=8)
    plot_fit(X.min(), X.max(), mu, sigma, theta, power)
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.title('Polynomial Regression Fit (lambda = %f)' % plambda)
    plt.show()
     
    # learning curve for polynomial fit
    X0 = c_[ones([m, 1]), Xpoly]
    X1 =  c_[ones([size(Xval_poly, 0), 1]), Xval_poly]
    error_train, error_val = learningCurve(X0, y, X1, yval, plambda)
    
    plt.plot(arange(1,m+1),error_train[1:m+1],'r', arange(1,m+1), error_val[1:m+1],'b')
    plt.title('Learning curve for linear regression')
    plt.legend(('Train', 'Cross Validation'),numpoints=2)
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.axis([0, 13, 0, 150])
    plt.show()
    
    print('# Training Examples\tTrain Error\tCross Validation Error')
    for i in range(1,m+1):
        print('  \t%d\t\t%f\t%f' % (i, error_train[i], error_val[i]))
    
    raw_input('\nProgram paused 7. Press any key to continue\n')
    
    ## =========== Part 8: Validation for Selecting Lambda =============
    #  You will now implement validationCurve to test various values of 
    #  lambda on a validation set. You will then use this to select the
    #  "best" lambda value.
    #

    lambda_vec, error_train, error_val = validationCurve(Xpoly, y, Xval_poly, yval)
    plt.plot(lambda_vec, error_train,'r', lambda_vec, error_val,'b')
    plt.legend(('Train', 'Cross Validation'),numpoints=2)
    plt.xlabel('lambda')
    plt.ylabel('Error')
    plt.show()

    print('lambda\t\tTrain Error\tValidation Error\n')
    for i in range(1,size(lambda_vec)):
        print('  \t%d\t\t%f\t%f' % (lambda_vec[i], error_train[i], error_val[i]))
