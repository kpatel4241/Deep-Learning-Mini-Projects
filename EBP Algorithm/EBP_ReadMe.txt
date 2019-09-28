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



Here we'll only consider a simple network with one hidden layer and one output unit. 
Here's the general algorithm for updating the weights with backpropagation:

    1.Set the weight steps for each layer to zero
        The input to hidden weights Δwij=0
        The hidden to output weights ΔWj=0

    2.For each record in the training data:

        Make a forward pass through the network, calculating the output y^(predict output)

        Calculate the error gradient in the output unit, δo=(y−y^)f′(z) where z=∑jWj*aj*z , the input to the output unit.

        Propagate the errors to the hidden layer δjh=δoWjf′(hj)

        Update the weight steps:
            ΔWj=ΔWj+δo*aj
            Δwij=Δwij+δjh*ai

    Update the weights, where η is the learning rate and m is the number of records:
        Wj=Wj+η*ΔWj
        wij=wij+η*Δwij

    Repeat for e epochs.


Now we're going to implement the backprop algorithm for a network trained on the graduate school admission data. 
We should have everything ew need from the previous exercises to complete this one.

Goals here:

    Implement the forward pass.
    Implement the backpropagation algorithm.
    Update the weights.
