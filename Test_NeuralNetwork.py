'''
Created on 27/01/2014

@author: pgiraldez
'''
# Load Python libraries
from numpy import *
from NeuralNetwork import *
from matplotlib import pyplot as plot
from scipy import optimize
import matplotlib.cm as cm
import scipy.io


def displayData(X):
    """
    Transforms each input row into a rectangular image part and plots
    the resulting image.
    """
    m, n = X.shape
    example_width = int(around(sqrt(n)))
    example_height = int(n / example_width)
    display_rows = int(sqrt(m))
    display_cols = int(m / display_rows)
    display_array = ones((
        display_rows * example_height, display_cols * example_width
    ))
    for i in range(display_rows):
        for j in range(display_cols):
            idx = i * display_cols + j
            image_part = X[idx, :].reshape((example_height, example_width))
            display_array[
                (j * example_height):((j + 1) * example_height),
                (i * example_width):((i + 1) * example_width)
            ] = image_part
    plot.imshow(display_array.T, cm.Greys)
    plot.show()




# Python __main___ module

if __name__ == '__main__':

    ## Setup the parameters you will use for this exercise
    input_layer_size  = 400  # 20x20 Input Images of Digits
    hidden_layer_size = 25   # 25 hidden units
    num_labels = 10          # 10 labels, from 1 to 10
    
    print('Loading and Visualizing Data ...\n')

    # Load the matlab file
    mat = scipy.io.loadmat('C:\Users\pgiraldez\Documents\Octave\mlclass-ex4\ex4data1.mat') 
    X = mat['X']
    y = mat['y'].flatten() 

    m = size(X, 0)
   
    # Randomly select 100 data points to display
    sel = random.permutation(size(X, 0))
    sel = sel[0:100]

    displayData(X[sel, :])

    raw_input('Press any key to continue\n')
    
    ## ================ Part 2: Loading Pameters ================
    # In this part of the exercise, we load some pre-initialized 
    # neural network parameters.

    print('\nLoading Saved Neural Network Parameters ...\n')

    # Load the weights into variables Theta1 and Theta2
    weights = scipy.io.loadmat('C:\Users\pgiraldez\Documents\Octave\mlclass-ex4\ex4weights.mat');
    Theta1 = weights['Theta1']
    Theta2 = weights['Theta2']
    
    # Unroll parameters
    nn_params = concatenate((Theta1.flatten(),Theta2.flatten()))
    
    ## ================ Part 3: Compute Cost Function ================
    # Feed-forward
    print('\nFeed-Forward using Neural Network\n')
    
    # Regualization parameter
    plambda = 0
    
    J, grad  = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, plambda)
    
    print('Cost at parameters (loaded from ex4weights): %f' % J)
    print('(this value should be about 0.287629)')
    
    ## ================ Part 4: Implement Regularization ================
    print('\nFeed-Forward using Neural Network w/regularization (lambda=1)\n')
    plambda = 1
    
    J, grad  = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, plambda)

    print('Cost at parameters (loaded from ex4weights): %f' % J)
    print('(this value should be about 0.383770)')

    ## ================ Part 5: Sigmoid Gradient ================
    print('\nEvaluating sigmoid gradient...')
    gradient = sigmoidGradient(array([1, -0.5, 0, 0.5, 1]))
    print('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:\n%s' % gradient)

    raw_input('\nPress any key to continue\n')

    ## ================ Part 6: Initializing Pameters ================
    print('\nInitializing Neural Network Parameters ...\n')

    init_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
    init_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

    # Unroll parameters
    initial_nn_params = concatenate([init_Theta1.flatten(), init_Theta2.flatten()]) 
    
    ## =============== Part 8: Implement Regularization ===============
    # Once your backpropagation implementation is correct, you should now
    # continue to implement the regularization with the cost and gradient.
    #

    print('Checking Backpropagation...')
    checkNNGradients()

    print('\nChecking Backpropagation (w/ Regularization) ... \n')

    #  Check gradients by running checkNNGradients
    plambda = 3
    checkNNGradients(plambda)

    # Also output the costFunction debugging values
    debug_J, debug_grad  = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, plambda)

    print
    print('\nCost at (fixed) debugging parameters (w/ lambda = 3): %f ', debug_J) 
    print('\n(this value should be about 0.576051)\n\n')
    # print shape(debug_grad)
    raw_input('\nPress any key to continue\n')

    ##=================== Part 8: Training NN ===================
    print('\nTraining Neural Network... \n')
    
    plambda = 1
    
    print('Comienza la optimizacion...\n')
    # print shape(initial_nn_params)
    
    result = optimize.minimize(
        nnCostFunction,
        initial_nn_params,
        args=(input_layer_size, hidden_layer_size, num_labels, X, y, plambda),
        method='CG',
        jac=True,
        options={'maxiter': 50,'disp': True}
    )

    #result = optimize.fmin_cg(nnCostFunction_Cost, initial_nn_params,fprime=nnCostFunction_Grad,args=(input_layer_size, hidden_layer_size, num_labels, X, y, plambda), maxiter=50, disp=1)
 
    print('Finaliza la optimizacion...\n')
    nn_params = result.x
    
    boundary = (input_layer_size + 1) * hidden_layer_size
    Theta1 = nn_params[:boundary].reshape([hidden_layer_size, input_layer_size + 1])
    Theta2 = nn_params[boundary:].reshape([num_labels, hidden_layer_size + 1])
    
    print('Visualizing Neural Network ...')
    displayData(Theta1[:, 1:])
    displayData(Theta2[:, 1:])
    predictions = predict(Theta1, Theta2, X)
    accuracy = 100 * mean(predictions == y)
    print('Training set accuracy: %0.2f %%' % accuracy)