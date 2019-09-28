GOAL
=====

The goal here is to predict if a student will be admitted to a graduate program based on these features. 
For this, we'll use a network with one output layer with one unit. We'll use a sigmoid function for the output unit activation. 


Data Preperation
=================

We actually need to transform the data first. 
The rank feature is categorical, the numbers don't encode any sort of relative values. 
Rank 2 is not twice as much as rank 1, rank 3 is not 1.5 more than rank 2. 
Instead, we need to use dummy variables to encode rank, splitting the data into four new columns encoded with ones or zeros. 
Rows with rank 1 have one in the rank 1 dummy column, and zeros in all other columns. Rows with rank 2 have one in the rank 2 dummy column, and zeros in all other columns. And so on.

We'll also need to standardize the GRE and GPA data, which means to scale the values such that they have zero mean and a standard deviation of 1. 
This is necessary because the sigmoid function squashes really small and really large inputs. 
The gradient of really small and large inputs is zero, which means that the gradient descent step will go to zero too. 
Since the GRE and GPA values are fairly large, we have to be really careful about how we initialize the weights or the gradient descent steps will die off and the network won't train. 
Instead, if we standardize the data, we can initialize the weights easily.

Now that the data is ready, we see that there are six input features: gre, gpa, and the four rank dummy variables.



Algorithm:
==========

Here's the general algorithm for updating the weights with gradient descent:

    1. Set the weight step to zero: Δwi=0

    2. For each record in the training data:
        Make a forward pass through the network, calculating the output y^=f(∑iwixi)
        
        Calculate the error term for the output unit, δ=(y−y^)∗f′(∑iwixi)
        
        Update the weight step Δwi=Δwi+δxi
    
    3.Update the weights wi=wi+ηΔwi  where η is the learning rate and m is the number of records. Here we're averaging the weight steps to help reduce any large variations in the training data.
    
    4.Repeat for eee epochs.


We're using the sigmoid for the activation function,
    f(h) = 1/(1+e^{-h})

And the gradient of the sigmoid is
    f′(h)=f(h)(1−f(h))

where h is the input to the output unit,
    h=∑iwixih


We'll implement gradient descent and train the network on the admissions data. 
Our goal here is to train the network until you reach a minimum in the mean square error (MSE) on the training set. 
We need to implement:

    The network output: output.
    The output error: error.
    The error term: error_term.
    Update the weight step: del_w +=.
    Update the weights: weights +=.

After We've written these parts, Run it in Spyder. 
The MSE will print out, as well as the accuracy on a test set, the fraction of correctly predicted admissions.

We can also tune our hyperparameters to get vary the MSE and Accuracy.