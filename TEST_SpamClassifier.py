'''
Created on 15/04/2014

@author: pgiraldez
'''

from numpy import *
from scipy.io import loadmat
from  nltk.stem import PorterStemmer
from sklearn import svm, grid_search
import re

def getVocabList():
    ## Read the fixed vocabulary list
    #fid = fopen('vocab.txt');
    fid = genfromtxt('C:/Users/pgiraldez/Documents/Octave/mlclass-ex6/vocab.txt',delimiter='\t')

    # Store all dictionary words in cell array vocab{}
    n = 1899     # Total number of words in the dictionary

    # For ease of implementation, we use a struct to map the strings => integers
    # In practice, you'll want to use some form of hashmap
    vocabList = []
    for i in open('C:/Users/pgiraldez/Documents/Octave/mlclass-ex6/vocab.txt'):
        vocabList.append( i.split()) 

    return vocabList

def processEmail(email_contents):
    #   word_indices = PROCESSEMAIL(email_contents) preprocesses 
    #   the body of an email and returns a list of indices of the 
    #   words contained in the email. 
    #
    
    # Load Vocabulary
    vocabList = getVocabList()

    # Init return value
    word_indices = array([])
    
    # Lower case
    email_contents = email_contents.lower()

    # Strip all HTML
    # Looks for any expression that starts with < and ends with > and replace
    # and does not have any < or > in the tag it with a space
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)

    # Handle Numbers
    # Look for one or more characters between 0-9
    email_contents = re.sub('[0-9]+', 'number', email_contents)

    # Handle URLS
    # Look for strings starting with http:// or https://
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)
    
    # Handle Email Addresses
    # Look for strings with @ in the middle
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)

    # Handle $ sign
    email_contents = re.sub('[$]+', 'dollar', email_contents)
    
    # ========================== Tokenize Email ===========================

    # Output the email to screen as well
    print('\n==== Processed Email ====\n\n')

    # Process file
    l = 0
    stemmer=PorterStemmer()

#    while email_contents:
      
#        str1 = [str(x) for x in filter(None, re.split('[ @,$,/,#,.,-,:,&,*,+,=,[,],?,!,(,),{,},,,",>,_,<,;,%,,\n,\r]',email_contents))]
    str1 = [str(x) for x in filter(None, re.split('[ ,@,$,/,#,.,-,:,&,*,+,=,,?,!,(,),{,},,,",>,_,<,;,%,\n,\t,\r]',email_contents))]
        
    # Remove any non alphanumeric characters
    str1 = [re.sub('[^a-zA-Z0-9]', '', x) for x in filter(None, str1)]
        
    # Stem the word 
    str1 = [stemmer.stem(kw) for kw in str1]
        
    for x in filter(None,str1):        
        # Skip the word if it is too short
        if len(x) < 2:
            continue

        # Look up the word in the dictionary and add to word_indices if
        # found
        
        for i in range( 1,len(vocabList)):
            str2 = vocabList[i][1]
            if cmp(x,str2) == 0:
                word_indices = hstack((word_indices, i))
                continue
        
        # Print to screen, ensuring that the output lines are not too long
        if (l + len(x) + 1) > 78:
            print
            l = 0

        print(' %s' %x),
        l = l + len(x) + 1

    # Print footer
    print('\n\n=========================\n');
    
    return word_indices

 
def emailFeatures(word_indices):
    #  Takes in a word_indices vector and 
    #    produces a feature vector from the word indices.
    
    # Total number of words in the dictionary
    n = 1899
 
    # Number of features    
    x = zeros([n, 1])

    for i in range(len(word_indices)):
        x[word_indices[i]] = 1
        
    return x
   
    
    
if __name__ == '__main__':
    
    ## ==================== Part 1: Email Preprocessing ====================
    #  To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
    #  to convert each email into a vector of features. In this part, you will
    #  implement the preprocessing steps for each email. You should
    #  complete the code in processEmail.m to produce a word indices vector
    #%  for a given email.

    print('\nPreprocessing sample email (emailSample1.txt)\n')
    
    # Extract Features
    file_contents = open('C:\Users\pgiraldez\Documents\Octave\mlclass-ex6\emailSample1.txt')
    email_contents= file_contents.read()
    word_indices  = processEmail(email_contents)
    
    # Print Stats
    print('Word Indices: \n')
    print(word_indices)
    print('\n\n')

    raw_input('Program paused 1. Press any key to continue\n')
    
    ##==================== Part 2: Feature Extraction ====================
    #  Now, you will convert each email into a vector of features in R^n. 
    #  You should complete the code in emailFeatures.m to produce a feature
    #  vector for a given email.

    print('\nExtracting features from sample email (emailSample1.txt)\n')

    # Extract Features
    file_contents = open('C:\Users\pgiraldez\Documents\Octave\mlclass-ex6\emailSample1.txt')
    email_contents= file_contents.read()
    word_indices  = processEmail(email_contents)
    features      = emailFeatures(word_indices)

    #Print Stats
    print('Length of feature vector: %d' % len(features))
    print('Number of non-zero entries: %d' % sum(features > 0))

    raw_input('\nProgram paused 2. Press any key to continue\n')
    
    ## =========== Part 3: Train Linear SVM for Spam Classification ========
    #  In this section, you will train a linear classifier to determine if an
    #  email is Spam or Not-Spam.

    # Load the Spam Email dataset
    # You will have X, y in your environment
    data = loadmat('C:\Users\pgiraldez\Documents\Octave\mlclass-ex6\spamTrain.mat') 
    X     = data['X']
    y     = int_(data['y']).flatten() 

    print('Training Linear SVM (Spam Classification)')
    print('(this may take 1 to 2 minutes) ...')

    C = 0.1
#     Y = copy(y)
#     Y[Y==0] = -1
#     Y = Y.flatten()
#    model = svmTrain(X, y, C, @linearKernel)
    model = svm.SVC(kernel='linear', C=10)     # gamma = 1/ sigma
    model.fit(X,y)

    p = model.predict(X)
    
    ta = mean(double(p == y))*100
    print('Training Accuracy: %f' % ta)

    raw_input('\nProgram paused 3. Press any key to continue\n')
    
    
    ## =================== Part 4: Test Spam Classification ================
    #  After training the classifier, we can evaluate it on a test set. We have
    #  included a test set in spamTest.mat

    # Load the test dataset
    # You will have Xtest, ytest in your environment
    # load('spamTest.mat');
    data  = loadmat('C:\Users\pgiraldez\Documents\Octave\mlclass-ex6\spamTest.mat') 
    Xtest = data['Xtest']
    ytest = int_(data['ytest']).flatten() 

    print('\nEvaluating the trained Linear SVM on a test set ...\n')

    p = model.predict(Xtest)

    ta = mean(double(p == ytest)) * 100
    print('Test Accuracy: %f\n' % ta)

    raw_input('\nProgram paused 4. Press any key to continue\n')

    ## ================= Part 5: Top Predictors of Spam ====================
    #  Since the model we are training is a linear SVM, we can inspect the
    #  weights learned by the model to understand better how it is determining
    #  whether an email is spam or not. The following code finds the words with
    #  the highest weights in the classifier. Informally, the classifier
    #  'thinks' that these words are the most likely indicators of spam.
    #

    # Sort the weights and obtin the vocabulary list
    coefList = (model.coef_).flatten()
    weight = sorted(coefList,reverse=True)
    vocabList = getVocabList()

    print('\nTop predictors of spam: \n')
    for i in range(15):
        print(' %-15s (%f) ' % (vocabList[list(coefList).index(weight[i])][1], weight[i]))

    print('\n\n')
    print('\nProgram paused. Press enter to continue.\n')
    raw_input('\nProgram paused 5. Press any key to continue\n')
