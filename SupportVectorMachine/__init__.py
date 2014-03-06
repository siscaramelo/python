#!/usr/bin/env python
'''
Created on 05/03/2014

@author: pabgir
'''
from numpy import *
from numpy import newaxis, r_, c_, mat, e
from numpy.linalg import norm

class model():
    X = None
    y = None
    kernelFunction = None
    alphas = None
    w = None

def svmTrain(X, Y, C, kernelFunction, tol = 1e-3, max_passes = 20):
    # Data parameters
    m = size(X, 0)
    n = size(X, 1)

    # Map 0 to -1
    Y = copy(Y)
    Y[Y==0] = -1
    Y = reshape(Y,[size(Y),1])

    # Variables
    alphas = zeros([m, 1])
    b      = 0
    E      = zeros([m, 1])
    passes = 0
    eta    = 0
    L      = 0
    H      = 0
    
    functionName = getattr(kernelFunction,'__name__')
    if (functionName == 'linearKernel'):
        # Vectorized computation for the Linear Kernel
        # This is equivalent to computing the kernel on every pair of examples
        K = X.dot(X.T)
        
    elif (functionName == 'gaussianKernel'):
        # Vectorized RBF Kernel
        # This is equivalent to computing the kernel on every pair of examples
        X2 = sum(X**2, 1)
        K = X2 + (X2.T - 2 * (X.dot(X.T)))
        K = kernelFunction(1, 0) ** K
        
    else:
        #    Pre-compute the Kernel Matrix
        #    The following can be slow due to the lack of vectorization       
        K = zeros(m)
        for i in range(1, m):
            for j in range(i, m):
                K[i, j] = kernelFunction(X[i,:].T, X[j,:].T)
                K[j, i] = K[i, j]     # the matrix is symmetric    
         
    # Train
    print('\nTraining ...')
    dots = 12
    while (passes < max_passes):
            
        num_changed_alphas = 0
        for i in range(1, m):        
            # Calculate Ei = f(x[i]) - y[i] using (2). 
            # E[i] = b + sum (X(i, :) * (repmat(alphas.*Y,1,n).*X)') - Y[i];
            E[i] = b + sum (alphas * Y * K[:,i]) - Y[i]
        
            if ((Y[i]*E[i] < -tol and alphas[i] < C) or (Y[i]*E[i] > tol and alphas[i] > 0)):
            
                # In practice, there are many heuristics one can use to select
                # the i and j. In this simplified code, we select them randomly.
                j = ceil((m-1) * random.random())
                while j == i:  # Make sure i \neq j
                    j = ceil((m-1) * random.random())
        

                # Calculate Ej = f(x[j]) - y[j] using (2).
                E[j] = b + sum(alphas * Y * K[:,j]) - Y[j]

                # Save old alphas
                alpha_i_old = alphas[i]
                alpha_j_old = alphas[j]
            
                # Compute L and H by (10) or (11). 
                if (Y[i] == Y[j]):
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                else:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                     
                if (L == H):
                    # continue to next i. 
                    continue
            
                # Compute eta by (14).
                eta = 2 * K[i,j] - K[i,i] - K[j,j]
                if (eta >= 0):
                    # continue to next i. 
                    continue
            
                # Compute and clip new value for alpha j using (12) and (15).
                alphas[j] = alphas[j] - (Y[j] * (E[i] - E[j])) / eta
            
                # Clip
                alphas[j] = min(H, alphas[j])
                alphas[j] = max(L, alphas[j])
            
                # Check if change in alpha is significant
                if (abs(alphas[j] - alpha_j_old) < tol):
                    # continue to next i. 
                    # replace anyway
                    alphas[j] = alpha_j_old
                    continue
            
                # Determine value for alpha i using (16). 
                alphas[i] = alphas[i] + Y[i]*Y[j]*(alpha_j_old - alphas[j])
            
                # Compute b1 and b2 using (17) and (18) respectively. 
                b1 = b - E[i] - Y[i] * (alphas[i] - alpha_i_old) *  K[i,j].T - Y[j] * (alphas[j] - alpha_j_old) * K[i,j].T
                b2 = b - E[j] - Y[i] * (alphas[i] - alpha_i_old) *  K[i,j].T - Y[j] * (alphas[j] - alpha_j_old) * K(j,j).T

                # Compute b by (19). 
                if (0 < alphas[i] and alphas[i] < C):
                    b = b1
                elif (0 < alphas[j] and alphas[j] < C):
                    b = b2
                else:
                    b = (b1+b2)/2

                num_changed_alphas = num_changed_alphas + 1
  
        if (num_changed_alphas == 0):
            passes = passes + 1
        else:
            passes = 0

        print('.')
        dots = dots + 1
        if dots > 78:
            dots = 0
            print('\n')

    print(' Done! \n\n')

    # Save the model
    idx = alphas > 0
    
    Xr= X[idx,:]
    yr= Y[idx]
    alphasr= alphas[idx]
    a1 = alphas*Y
    a2 = a1.T.dot(X)
    wr = a2.T
    
    model.X= Xr
    model.y= yr
    model.kernelFunction = kernelFunction
    model.b= b
    model.alphas= alphasr
    model.w = wr

    return model
    #return Xr, yr, kernelFunction, b, alphasr, wr

def linearKernel(x1, x2):
    #  linearKernel(x1, x2) returns a linear kernel between x1 and x2
    #  and returns the value in sim
    # 
    
    # Ensure that x1 and x2 are column vectors
    x1 = reshape(x1,[1,size(x1)])
    x2 = reshape(x2,[1,size(x2)])

    # Compute the kernel
    sim = x1.T.dot(x2)      # dot product
    
    return sim

def gaussianKernel(x1, x2, sigma):
    #   gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
    #   and returns the value in sim

    # Ensure that x1 and x2 are column vectors
    x1 = reshape(x1,[size(x1),1])
    x2 = reshape(x2,[size(x2),1])

    sim = 0.
    #  This function to return the similarity between x1
    # and x2 computed using a Gaussian kernel with bandwidth
    # sigma
    # 
    dif = (x1-x2).T
    sim = exp(-dif.dot(dif.T)/(2.*sigma**2))
    
    return sim
