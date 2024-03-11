'''naive_bayes_multinomial.py
Naive Bayes classifier with Multinomial likelihood for discrete features
Maya Purohit 
CS 251: Data Analysis and Visualization
Fall 2023
'''
import numpy as np


class NaiveBayes:
    '''Naive Bayes classifier using Multinomial likeilihoods (discrete data belonging to any
     number of classes)'''
    def __init__(self, num_classes):
        '''Naive Bayes constructor

        TODO:
        - Add instance variable for `num_classes`.
        - Add placeholder instance variables the class prior probabilities and class likelihoods (assigned to None).
        You may store the priors and likelihoods themselves or the logs of them. Be sure to use variable names that make
        clear your choice of which version you are maintaining.
        '''
        self.num_classes = num_classes

        # class_priors: ndarray. shape=(num_classes,).
        #   Probability that a training example belongs to each of the classes
        #   For spam filter: prob training example is spam or ham
        self.class_priors = None
        # class_likelihoods: ndarray. shape=(num_classes, num_features).
        #   Probability that each word appears within class c
        self.class_likelihoods = None

    def get_priors(self):
        '''Returns the class priors (or log of class priors if storing that)'''
        return self.class_priors
        

    def get_likelihoods(self):
        '''Returns the class likelihoods (or log of class likelihoods if storing that)'''
        return self.class_likelihoods

    def train(self, data, y):
        '''Train the Naive Bayes classifier so that it records the "statistics" of the training set:
        class priors (i.e. how likely an email is in the training set to be spam or ham?) and the
        class likelihoods (the probability of a word appearing in each class — spam or ham)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.

        TODO:
        - Compute the class priors and class likelihoods (i.e. your instance variables) that are needed for
        Bayes Rule. See equations in notebook.
        '''

        unique, counts = np.unique(y, return_counts= True) #find the number of ham and spam emails in the training set 
        likelihoods = np.zeros((self.num_classes, data.shape[1]))
        self.class_priors = counts/data.shape[0] #finds the probability of having a spam or ham email 
        totalData = np.hstack((data, np.reshape(y, (y.shape[0], 1)))) #stacks the classes and the data into one numpy array 
        for i in range(self.num_classes): #loop through the types of classes 
            partData = totalData[totalData[:, -1] == i] #extracted all of the data with a certain class 
            likelihoods[i, :] = (np.sum(partData[:, :-1], axis = 0) + 1) /(np.sum(partData[:, :-1]) + data.shape[1]) # finds the likelihood of each word in top words to appear in a ham or spam email 
        self.class_likelihoods = likelihoods #sets this as the likelihood 


    def predict(self, data):
        '''Combine the class likelihoods and priors to compute the posterior distribution. The
        predicted class for a test sample from `data` is the class that yields the highest posterior
        probability.

        Parameters:
        -----------
        data: ndarray. shape=(num_test_samps, num_features). Data to predict the class of
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_samps,). Predicted class of each test data sample.

        TODO:
        - For the test samples, we want to compute the log of the posterior by evaluating
        the the log of the right-hand side of Bayes Rule without the denominator (see notebook for
        equation). This can be done without loops.
        - Predict the class of each test sample according to the class that produces the largest
        log(posterior) probability (hint: this can also be done without loops).

        NOTE: Remember that you are computing the LOG of the posterior (see notebook for equation).
        NOTE: The argmax function could be useful here.
        '''

        probs = np.log(self.class_priors) + data @ np.log(self.class_likelihoods.T) #finf the log of the likelihood and priors and multiply the data we get to get probabilities 
        preds = np.argmax(probs, axis = 1) #finds the index of the maximum probability in each of the rows and replaces it, getting the number of the class 

        return preds



    def accuracy(self, y, y_pred):
        '''Computes accuracy based on percent correct: Proportion of predicted class labels `y_pred`
        that match the true values `y`.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_sams,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_sams,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        float. Between 0 and 1. Proportion correct classification.

        NOTE: Can be done without any loops
        '''

        subtract = y_pred - y
        numCorrect = subtract[subtract == 0].shape[0] #subtract all of the values in y and y_pred, if they are the same, then the class was guessed correctly 

        return (numCorrect/y.shape[0])
    

    def confusion_matrix(self, y, y_pred):
        '''Create a confusion matrix based on the ground truth class labels (`y`) and those predicted
        by the classifier (`y_pred`).

        Recall: the rows represent the "actual" ground truth labels, the columns represent the
        predicted labels.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_samps,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_samps,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        ndarray. shape=(num_classes, num_classes).
            Confusion matrix
        '''
        confusion_matrix = np.zeros((self.num_classes, self.num_classes)) #initialize a matrix with zeros 

        for i in range(len(y)):
            confusion_matrix[int(y[i]), int(y_pred[i])] += 1 #go through each index in the predictions and actual values and add 1 in the confusion matrix at the location determined by the class of the actual and predicted value 
            
        #row is the actual value and columns are the predicted values 

        return confusion_matrix
