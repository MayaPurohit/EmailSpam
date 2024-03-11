'''knn.py
K-Nearest Neighbors algorithm for classification
Maya Purohit
CS 251: Data Analysis and Visualization
Fall 2023
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class KNN:
    '''K-Nearest Neighbors supervised learning algorithm'''
    def __init__(self, num_classes):
        '''KNN constructor

        TODO:
        - Add instance variable for `num_classes`
        '''
        self.num_classes = num_classes

        # exemplars: ndarray. shape=(num_train_samps, num_features).
        #   Memorized training examples
        self.exemplars = None
        # classes: ndarray. shape=(num_train_samps,).
        #   Classes of memorized training examples
        self.classes = None

    def train(self, data, y):
        '''Train the KNN classifier on the data `data`, where training samples have corresponding
        class labels in `y`.

        Parameters:
        -----------
        data: ndarray. shape=(num_train_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_train_samps,). Corresponding class of each data sample.

        TODO:
        - Set the `exemplars` and `classes` instance variables such that the classifier memorizes
        the training data.
        '''
        self.exemplars = data #memorization of information
        self.classes = y

        pass

    def euclidean(self, pt_1, exemplars):
        '''Compute the Euclidean distance between data sample `pt_1` and and all exemplars


        Parameters:
        -----------
        pt1: ndarray. shape=(num_features,)
        exemplars: ndarray. shape=(num_train_samps, num_features)

        Returns:
        -----------
        ndarray. shape=(num_train_samps,).'''

        dist = np.sum(np.square(exemplars - pt_1), axis =1) #finds the euclidian distance 
        squareR = np.sqrt(dist)
        return squareR #returns the distances 
    
    def manhattan(self, pt_1, exemplars):
        '''Compute the manhattan distance between data sample `pt_1` and and all exemplars
    

        Parameters:
        -----------
        pt1: ndarray. shape=(num_features,)
        exemplars: ndarray. shape=(num_train_samps, num_features)

        Returns:
        -----------
        ndarray. shape=(num_train_samps,).'''

        dist = np.sum((exemplars - pt_1), axis =1) #finds the euclidian distance 
        return dist #returns the distances 

    def predict(self, data, k, euclidean = True):
        '''Use the trained KNN classifier to predict the class label of each test sample in `data`.
        Determine class by voting: find the closest `k` training exemplars (training samples) and
        the class is the majority vote of the classes of these training exemplars.

        Parameters:
        -----------
        data: ndarray. shape=(num_test_samps, num_features). Data to predict the class of
            Need not be the data used to train the network.
        k: int. Determines the neighborhood size of training points around each test sample used to
            make class predictions. In other words, how many training samples vote to determine the
            predicted class of a nearby test sample.

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_test_samps,). Predicted class of each test data
        sample.

        TODO:
        - Compute the distance from each test sample to all the training exemplars.
        - Among the closest `k` training exemplars to each test sample, count up how many belong
        to which class.
        - The predicted class of the test sample is the majority vote.
        '''
        predClasses = []
        for i in range(data.shape[0]):
            pt1 = data[i, :] #gets each points
            if(euclidean == False): #Extension to see difference between two forms of distance
                orDist = self.manhattan(pt1, self.exemplars)  #finds its manhattan distance from each exemplar
            else:
                orDist = self.euclidean(pt1, self.exemplars) #finds its euclidean distance from each exemplar

            indexes = np.argsort(orDist) #sort the distances from lowest to highest 
            orClasses = self.classes[indexes] #order the classes the same way 
            orClasses = orClasses[:k] #only take the first few classes
            unique, counts = np.unique(orClasses, return_counts = True) #get the unique unique classes and the counts of each class
            maxIn = np.argmax(counts) #finds the index with the highest count
            predClasses.append(unique[maxIn]) #finds the class at the index with the highest count 

            

        return np.array(predClasses)

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

        subtract = y_pred - y #in the subtracted matrix, 0 will appear if the correct class is predicted
        numCorrect = subtract[subtract == 0].shape[0] #we can find the number of 0's by using logical indexing 

        return (numCorrect/y.shape[0]) #we can divide by the total number of samples to get the accuracy 

        

    def plot_predictions(self, k, n_sample_pts):
        '''Paints the data space in colors corresponding to which class the classifier would
         hypothetically assign to data samples appearing in each region.

        Parameters:
        -----------
        k: int. Determines the neighborhood size of training points around each test sample used to
            make class predictions. In other words, how many training samples vote to determine the
            predicted class of a nearby test sample.
        n_sample_pts: int.
            How many points to divide up the input data space into along the x and y axes to plug
            into KNN at which we are determining the predicted class. Think of this as regularly
            spaced 2D "fake data" that we generate and plug into KNN and get predictions at.

        TODO:
        - Pick a discrete/qualitative color scheme. We suggest, like in the clustering project, to
        use either the Okabe & Ito or one of the Petroff color palettes: https://github.com/proplot-dev/proplot/issues/424
        - Wrap your colors list as a `ListedColormap` object (already imported above) so that matplotlib can parse it.
        - Make an ndarray of length `n_sample_pts` of regularly spaced points between -40 and +40.
        - Call `np.meshgrid` on your sampling vector to get the x and y coordinates of your 2D
        "fake data" sample points in the square region from [-40, 40] to [40, 40].
            - Example: x, y = np.meshgrid(samp_vec, samp_vec)
        - Combine your `x` and `y` sample coordinates into a single ndarray and reshape it so that
        you can plug it in as your `data` in self.predict.
            - Shape of `x` should be (n_sample_pts, n_sample_pts). You want to make your input to
            self.predict of shape=(n_sample_pts*n_sample_pts, 2).
        - Reshape the predicted classes (`y_pred`) in a square grid format for plotting in 2D.
        shape=(n_sample_pts, n_sample_pts).
        - Use the `plt.pcolormesh` function to create your plot. Use the `cmap` optional parameter
        to specify your discrete ColorBrewer color palette.
        - Add a colorbar to your plot


        
        '''

        gitColor = ['#F0E442', '#0072B2', '#D55E00', '#CC79A7']
 
        colors = ListedColormap(gitColor) #makes a color map 
        samp_vec = np.linspace(-40, 40, n_sample_pts)
        x_samples, y_samples = np.meshgrid(samp_vec, samp_vec) #create num_sample_pts by num_sample_pts grid
        x_flatten = x_samples.flatten()
        x_flatten = np.reshape(x_flatten, (x_flatten.shape[0], 1))

        y_flatten = y_samples.flatten()
        y_flatten = np.reshape(y_flatten, (y_flatten.shape[0], 1))  
        xy_stack = np.hstack((x_flatten, y_flatten))  # Nx2 flatten the grids and stack them together to make predictions on 
        
        predictions = self.predict(data = xy_stack, k = k)  # N, #make predictions on each point in the graph
        predictions = np.reshape(predictions, (n_sample_pts, n_sample_pts)) #reshape the matrix 
        plt.pcolormesh(x_samples, y_samples, predictions, cmap = colors) #make a graph with a color bar 
        plt.title("Predicted Classes")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.colorbar()



        


        
        pass

    def confusion_matrix(self, y, y_pred):
        '''Create a confusion matrix based on the ground truth class labels (`y`) and those predicted
        by the classifier (`y_pred`).

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
        # To get the number of classes, you can use the np.unique
        # function to identify the number of unique categories in the
        # y matrix.

        confusion_matrix = np.zeros((self.num_classes, self.num_classes))

        for i in range(len(y)):
            confusion_matrix[int(y[i]), int(y_pred[i])] += 1 #go through each index in the predictions and actual values and add 1 in the confusion matrix at the location determined by the class of the actual and predicted value 

        #row is the actual value and columns are the predicted values 

        return confusion_matrix
        
