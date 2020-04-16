

# A Logistic Regression algorithm with regularized weights...

from classifier import *

#Note: Here the bias term is considered as the last added feature 
def log_inf(x):
    if not isinstance(x, (list, tuple)):
        #print("FIRST")
        return np.log(x) if x>0 else float('Inf')
    else:
        #print("SECOND")
        Result=[]
        for i in range(len(x)):
            Result.append(np.log(x[i]) if x[i]>0 else float('Inf'))
        return Result
class LogisticRegression(Classifier):
    ''' Implements the LogisticRegression For Classification... '''
    def __init__(self, lembda=0.001):        
        """
            lembda= Regularization parameter...            
        """
        Classifier.__init__(self,lembda)
        
        pass
    def sigmoid(self,z):
        """
            Compute the sigmoid function 
            Input:
                z can be a scalar or a matrix
            Returns:
                sigmoid of the input variable z
        """

        # Your Code here
        return 1.0/(1+np.exp(-z))
    
    def hypothesis(self, X, theta):
        '''
            Computes the hypothesis for over given input examples (X) and parameters (thetas).

            Input:
                X: can be either a single n X d-dimensional vector or n X d dimensional matrix
                theta: Must be a d-dimensional vector
            Return:
                The computed hypothesis
        '''
        
        # Your Code here

        return self.sigmoid(np.dot(X,theta))


    def cost_function(self, X, Y, theta):
        '''
            Computes the Cost function for given input data (X) and labels (Y).

            Input:
                X: can be either a single n X d-dimensional vector or n X d dimensional matrix of inputs
                theata: must  d X 1-dimensional vector for representing vectors
                Y: Must be n X 1-dimensional label vector
                
            Return:
                Returns the cost of hypothesis with input parameters 
        '''
    
        # Your Code here
        Hypothesis=self.hypothesis(X,theta)
        return (np.sum((-Y*np.log(Hypothesis )) - ((1-Y)*np.log(list(1-Hypothesis))))/len(X))
        
    
    def derivative_cost_function(self, X, Y, thetas):
        '''
            Computes the derivates of Cost function w.r.t input parameters (thetas)  
            for given input and labels.

            Input:
            ------
                X: can be either a single n X d-dimensional vector or n X d dimensional matrix of inputs
                theata: must  d X 1-dimensional vector for representing vectors
                Y: Must be n X 1-dimensional label vector
            Returns:
            ------
                partial_thetas: a d X 1-dimensional vector of partial derivatives of cost function w.r.t parameters..
        '''
        
        # Your Code here
        new_theetas=[]
        for i in range (len(thetas)):
            if i+1!=len(thetas):
                new_theetas.append(np.sum(np.dot(X[:,i],self.hypothesis(X,thetas)-Y))/len(X) + self.lembda*thetas[i])
            else:
                new_theetas.append(np.sum(self.hypothesis(X,thetas)-Y)/len(X)  + self.lembda*thetas[i])
        return np.array(new_theetas)
    
    def train(self, X, Y, optimizer):
        ''' Train classifier using the given 
            X [m x d] data matrix and Y labels matrix
            
            Input:
            ------
            X: [m x d] a data matrix of m d-dimensional examples.
            Y: [m x 1] a label vector.
            optimizer: an object of Optimizer class, used to find
                       find best set of parameters by calling its
                       gradient descent method...
            Returns:
            -----------
            Nothing
            '''
        
        # Your Code here 
        # Use optimizer here
        self.theta=optimizer.gradient_descent(X,Y,self.cost_function,self.derivative_cost_function)

        
    def predict(self, X):
        
        """
        Test the trained perceptron classifier result on the given examples X
        
                   
            Input:
            ------
            X: [m x d] a matrix of m  d-dimensional test examples.
           
            Returns:
            -----------
                pclass: the predicted class for the given set of examples, i.e. to which it belongs
        """
        
        num_test = X.shape[0]
        
        # Your Code here
        Results=self.hypothesis(X,self.theta)
        Output=[]
        for i in range(num_test):
            if Results[i]<0.5:
                Output.append([int(0)])
            else:
                Output.append([int(1)])
        return np.array(Output)
