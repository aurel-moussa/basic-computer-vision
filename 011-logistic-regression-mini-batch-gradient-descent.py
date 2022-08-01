#Learning how to do logistic regression with mini-batch gradient descent
#Representing data as a Dataset obeject, usign PyTorch to create log regression model,
#setting criterion to calculate loss, creating a data loader, setting batch size,
#creating an optimizer function to update the model parameters and set the learning rate,
#and then traaaaain the model

#IMPORT LIBRARIES
#First, let's get all the good stuff
import numpy as np #use arrays to manipulate and store data
import matplotlib.pyplot as plt #for graphing data and loss curves
from mpl_toolkits import mplot3d #for graphing data and loss curves

import torch # PyTorch Library
from torch.utils.data import Dataset, DataLoader #help create the dataset and perform mini-batch
import torch.nn as nn # PyTorch Neural Network

#HELPER FUNCTIONS
# Create class for plotting and the function for plotting

class plot_error_surfaces(object):
    
    # Construstor
    def __init__(self, w_range, b_range, X, Y, n_samples = 30, go = True):
        W = np.linspace(-w_range, w_range, n_samples)
        B = np.linspace(-b_range, b_range, n_samples)
        w, b = np.meshgrid(W, B)    
        Z = np.zeros((30, 30))
        count1 = 0
        self.y = Y.numpy()
        self.x = X.numpy()
        for w1, b1 in zip(w, b):
            count2 = 0
            for w2, b2 in zip(w1, b1):
                yhat= 1 / (1 + np.exp(-1*(w2*self.x+b2)))
                Z[count1,count2]=-1*np.mean(self.y*np.log(yhat+1e-16) +(1-self.y)*np.log(1-yhat+1e-16))
                count2 += 1   
            count1 += 1
        self.Z = Z
        self.w = w
        self.b = b
        self.W = []
        self.B = []
        self.LOSS = []
        self.n = 0
        if go == True:
            plt.figure()
            plt.figure(figsize=(7.5, 5))
            plt.axes(projection='3d').plot_surface(self.w, self.b, self.Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
            plt.title('Loss Surface')
            plt.xlabel('w')
            plt.ylabel('b')
            plt.show()
            plt.figure()
            plt.title('Loss Surface Contour')
            plt.xlabel('w')
            plt.ylabel('b')
            plt.contour(self.w, self.b, self.Z)
            plt.show()
            
     # Setter
    def set_para_loss(self, model, loss):
        self.n = self.n + 1
        self.W.append(list(model.parameters())[0].item())
        self.B.append(list(model.parameters())[1].item())
        self.LOSS.append(loss)
    
    # Plot diagram
    def final_plot(self): 
        ax = plt.axes(projection='3d')
        ax.plot_wireframe(self.w, self.b, self.Z)
        ax.scatter(self.W, self.B, self.LOSS, c='r', marker='x', s=200, alpha=1)
        plt.figure()
        plt.contour(self.w, self.b, self.Z)
        plt.scatter(self.W, self.B, c='r', marker='x')
        plt.xlabel('w')
        plt.ylabel('b')
        plt.show()
        
    # Plot diagram
    def plot_ps(self):
        plt.subplot(121)
        plt.ylim
        plt.plot(self.x[self.y==0], self.y[self.y==0], 'ro', label="training points")
        plt.plot(self.x[self.y==1], self.y[self.y==1]-1, 'o', label="training points")
        plt.plot(self.x, self.W[-1] * self.x + self.B[-1], label="estimated line")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim((-0.1, 2))
        plt.title('Data Space Iteration: ' + str(self.n))
        plt.show()
        plt.subplot(122)
        plt.contour(self.w, self.b, self.Z)
        plt.scatter(self.W, self.B, c='r', marker='x')
        plt.title('Loss Surface Contour Iteration' + str(self.n))
        plt.xlabel('w')
        plt.ylabel('b')
        
# Plot the diagram

def PlotStuff(X, Y, model, epoch, leg=True):
    
    plt.plot(X.numpy(), model(X).detach().numpy(), label=('epoch ' + str(epoch)))
    plt.plot(X.numpy(), Y.numpy(), 'r')
    if leg == True:
        plt.legend()
    else:
        pass

      
#CONFIGURATIONS
# Setting the random seed to control randomness and give reproducable results
torch.manual_seed(0)

#LOADING DATA
#There is an existing Dataset class, which we imported above from PyTorch
#We want to have a custom dataset that inherits most of the stuff, but override some methods, namely
#__len__ so that len(dataset) returns dataset size
#__getitem__ so that we can use indexung, e.g., dataset[2] to get the 2nd sample from the dataset

# Create the custom Data class which inherits Dataset
class Data(Dataset):
    
    # Constructor
    def __init__(self):
        # Create X values from -1 to 1 with step .1
        self.x = torch.arange(-1, 1, 0.1).view(-1, 1)
        # Create Y values all set to 0
        self.y = torch.zeros(self.x.shape[0], 1)
        # Set the X values above 0.2 to 1
        self.y[self.x[:, 0] > 0.2] = 1
        # Set the .len attribute because we need to override the __len__ method
        self.len = self.x.shape[0]
    
    # Getter that returns the data at the given index
    def __getitem__(self, index):      
        return self.x[index], self.y[index]
    
    # Get length of the dataset
    def __len__(self):
        return self.len

#Make a Data object and have a peek
data_set = Data()
data_set.x
data_set.y
len(data_set)
x,y = data_set[0]
print("x = {},  y = {}".format(x,y))
x,y = data_set[1]
print("x = {},  y = {}".format(x,y))

#For visual purposes, we can seprate this one-dimensional dataset into two classes
plt.plot(data_set.x[data_set.y==0], data_set.y[data_set.y==0], 'ro', label="y=0")
plt.plot(data_set.x[data_set.y==1], data_set.y[data_set.y==1]-1, 'o', label="y=1")
plt.xlabel('x')
plt.legend()          

#CREATE MODEL AND TOTAL LOSS / COST FUNCTION
#We're now going to create a logistic regression model using PyTorch
#Using Scikit-Learn would be more common, because it is easier to use and set-up for (simpler) Machine Learning functions
#PyTorch is more used for Deep Learning functions

#Logistic Regression has a single layer, where input is the number of features an X value of the dataset has (how many X dimensions)
#With a single output layer. The output of this layer is then pushed into a sigmoid function, in order to get an output form that between 0 and 1
#Have a look at a sigmoid curve to see that
#Using the sigmoid ffunction allows us to turn the output of the layer into a classifcation problem, where 1 is one class, and 0 is the other class

#BUILDING THE LOGISTIC REGRESISON CLASS
# Create logistic_regression class that inherits nn.Module which is the base class for all neural networks
class logistic_regression(nn.Module):
    
    # Constructor
    def __init__(self, n_inputs):
        super(logistic_regression, self).__init__()
        # Single layer of Logistic Regression with number of inputs being n_inputs and there being 1 output 
        self.linear = nn.Linear(n_inputs, 1) #The class we build should only have 1 output layer
        
    # Prediction
    def forward(self, x):
        # Using the input x value puts it through the single layer defined above then puts the output through the sigmoid function and returns the result
        yhat = torch.sigmoid(self.linear(x))
        return yhat #our prediction as to what y should be according to our model y-hat

#hecking the number of features that X has (size of input)      
x,y = data_set[0]
len(x)

# Create the logistic_regression result
model = logistic_regression(1)

#make a prediction sigma  𝜎  using the forward function defined above
x = torch.tensor([-1.0])
sigma = model(x)
sigma

#using our own data to make a prediction
x,y = data_set[2]
sigma = model(x)
sigma

#PLOT_ERROR_SURFACES object to visualize the data space and learnabel parameters space for use during training
#This allows us to tsee a Loss Surface graph, the loss values varying across w and b (yellow is high loss, dark blue is low loss i.e. the good place)
#Loss Surface contour is a top-down view of Loss Surface graph

# Create the plot_error_surfaces object
# 15 is the range of w
# 13 is the range of b
# data_set[:][0] are all the X values
# data_set[:][1] are all the Y values

get_surface = plot_error_surfaces(15, 13, data_set[:][0], data_set[:][1])

#define a criteiron using binary cross entryop loss
#measures the loss between prediction and actual value
criterion = nn.BCELoss()

x, y = data_set[0]
print("x = {},  y = {}".format(x,y))
sigma = model(x)
sigma #our prediction for sigma
loss = criterion(sigma, y)
loss #the difference between our prediction and the actual y value

#SETTING BATCH SIZE USING DATA LOADER
#One has to use data loader in PyTorch that will output a batch of data, the input is the dataset and batch_size
batch_size=10
trainloader = DataLoader(dataset = data_set, batch_size = 10)
dataset_iter = iter(trainloader)
X,y=next(dataset_iter )

#SETTING LEARNING RATE
#We set the learning rate by setting it as a parameter in the optimizer along with the parameters of the logistic regression model we are training
#The job of the optimizer, torch.optim.SGD, is to use the loss generated by the criterion to update the model parameters according to the learning rate
#SGD stands for Stochastic Gradient Descent which typically means that the batch size is set to 1, but the data loader we set up above has turned this into Mini-Batch Gradient Descent.

learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

#TRAIN MODEL WITH MINI BATCH GRADIENT DESCENT
#Let us train the model using various batch sizes and learning rates
#Starting of with  batch size of the data loader to 5 and the number of epochs to 250

#reinitzing the get_surface object
get_surface = plot_error_surfaces(15, 13, data_set[:][0], data_set[:][1], 30)

#TRAIN THE MODEL
# First we create an instance of the model we want to train
model = logistic_regression(1)
# We create a criterion which will measure loss
criterion = nn.BCELoss()
# We create a data loader with the dataset and specified batch size of 5
trainloader = DataLoader(dataset = data_set, batch_size = 5)
# We create an optimizer with the model parameters and learning rate
optimizer = torch.optim.SGD(model.parameters(), lr = .01)
# Then we set the number of epochs which is the total number of times we will train on the entire training dataset
epochs=500
# This will store the loss over iterations so we can plot it at the end
loss_values = []

# Loop will execute for number of epochs
for epoch in range(epochs):
    # For each batch in the training data
    for x, y in trainloader:
        # Make our predictions from the X values
        yhat = model(x)
        # Measure the loss between our prediction and actual Y values
        loss = criterion(yhat, y)
        # Resets the calculated gradient value, this must be done each time as it accumulates if we do not reset
        optimizer.zero_grad()
        # Calculates the gradient value with respect to each weight and bias
        loss.backward()
        # Updates the weight and bias according to calculated gradient value
        optimizer.step()
        # Set the parameters for the loss surface contour graphs
        get_surface.set_para_loss(model, loss.tolist())
        # Saves the loss of the iteration
        loss_values.append(loss)
    # Want to print the Data Space for the current iteration every 20 epochs
    if epoch % 20 == 0:
        get_surface.plot_ps()
        

#Let us have a look at the final values of the WEIGHTs and BIAS        
w = model.state_dict()['linear.weight'].data[0]
b = model.state_dict()['linear.bias'].data[0]
print("w = ", w, "b = ", b)

#Accuracy prediction
# Getting the predictions
yhat = model(data_set.x)
# Rounding the prediction to the nearedt integer 0 or 1 representing the classes
yhat = torch.round(yhat)
# Counter to keep track of correct predictions
correct = 0
# Goes through each prediction and actual y value
for prediction, actual in zip(yhat, data_set.y):
    # Compares if the prediction and actualy y value are the same
    if (prediction == actual):
        # Adds to counter if prediction is correct
        correct+=1
# Outputs the accuracy by dividing the correct predictions by the length of the dataset
print("Accuracy: ", correct/len(data_set)*100, "%")

#Plotting Cost/Loss vs Iteration
plt.plot(loss_values)
plt.xlabel("Iteration")
plt.ylabel("Cost")

#STOCHAISTIC GRADIENT DESCENT
#Now, we will set batch of data loader to 1, so that gradient descent is performed for each example
#This is called Stoachsitic Gradient Descent
#Well set 100 epochs

#Reinitialze the get_surface object
get_surface = plot_error_surfaces(15, 13, data_set[:][0], data_set[:][1], 30)

#TRAIN
# First we create an instance of the model we want to train
model = logistic_regression(1)
# We create a criterion which will measure loss
criterion = nn.BCELoss()
# We create a data loader with the dataset and specified batch size of 1
trainloader = DataLoader(dataset = data_set, batch_size = 1)
# We create an optimizer with the model parameters and learning rate
optimizer = torch.optim.SGD(model.parameters(), lr = .01)
# Then we set the number of epochs which is the total number of times we will train on the entire training dataset
epochs=100
# This will store the loss over iterations so we can plot it at the end
loss_values = []

# Loop will execute for number of epochs
for epoch in range(epochs):
    # For each batch in the training data
    for x, y in trainloader:
        # Make our predictions from the X values
        yhat = model(x)
        # Measure the loss between our prediction and actual Y values
        loss = criterion(yhat, y)
        # Resets the calculated gradient value, this must be done each time as it accumulates if we do not reset
        optimizer.zero_grad()
        # Calculates the gradient value with respect to each weight and bias
        loss.backward()
        # Updates the weight and bias according to calculated gradient value
        optimizer.step()
        # Set the parameters for the loss surface contour graphs
        get_surface.set_para_loss(model, loss.tolist())
        # Saves the loss of the iteration
        loss_values.append(loss)
    # Want to print the Data Space for the current iteration every 20 epochs
    if epoch % 20 == 0:
        get_surface.plot_ps()
        
w = model.state_dict()['linear.weight'].data[0]
b = model.state_dict()['linear.bias'].data[0]
print("w = ", w, "b = ", b)

# Getting the predictions
yhat = model(data_set.x)
# Rounding the prediction to the nearedt integer 0 or 1 representing the classes
yhat = torch.round(yhat)
# Counter to keep track of correct predictions
correct = 0
# Goes through each prediction and actual y value
for prediction, actual in zip(yhat, data_set.y):
    # Compares if the prediction and actualy y value are the same
    if (prediction == actual):
        # Adds to counter if prediction is correct
        correct+=1
# Outputs the accuracy by dividing the correct predictions by the length of the dataset
print("Accuracy: ", correct/len(data_set)*100, "%")

plt.plot(loss_values)
plt.xlabel("Iteration")
plt.ylabel("Cost")

#HIGH LEARNING RATE
#We wil run this again, with a batch size of 1, and this time with a learning rate of 0.1 and observe what happens with this high rate
#Reinitilaize get_surfaces
get_surface = plot_error_surfaces(15, 13, data_set[:][0], data_set[:][1], 30)

# First we create an instance of the model we want to train
model = logistic_regression(1)
# We create a criterion that will measure loss
criterion = nn.BCELoss()
# We create a data loader with the dataset and specified batch size of 1
trainloader = DataLoader(dataset = data_set, batch_size = 1)
# We create an optimizer with the model parameters and learning rate
optimizer = torch.optim.SGD(model.parameters(), lr = 1)
# Then we set the number of epochs which is the total number of times we will train on the entire training dataset
epochs=100
# This will store the loss over iterations so we can plot it at the end
loss_values = []

# Loop will execute for number of epochs
for epoch in range(epochs):
    # For each batch in the training data
    for x, y in trainloader:
        # Make our predictions from the X values
        yhat = model(x)
        # Measure the loss between our prediction and actual Y values
        loss = criterion(yhat, y)
        # Resets the calculated gradient value, this must be done each time as it accumulates if we do not reset
        optimizer.zero_grad()
        # Calculates the gradient value with respect to each weight and bias
        loss.backward()
        # Updates the weight and bias according to calculated gradient value
        optimizer.step()
        # Set the parameters for the loss surface contour graphs
        get_surface.set_para_loss(model, loss.tolist())
        # Saves the loss of the iteration
        loss_values.append(loss)
    # Want to print the Data Space for the current iteration every 20 epochs
    if epoch % 20 == 0:
        get_surface.plot_ps()
        
#In above example due to the high learning rate the Loss Surface Contour graph 
#has increased movement over the previous example and also moves in multiple directions 
#due to the minimum being overshot.

w = model.state_dict()['linear.weight'].data[0]
b = model.state_dict()['linear.bias'].data[0]
print("w = ", w, "b = ", b)
#The weights and biases correspond to the above graphs

# Getting the predictions
yhat = model(data_set.x)
# Rounding the prediction to the nearedt integer 0 or 1 representing the classes
yhat = torch.round(yhat)
# Counter to keep track of correct predictions
correct = 0
# Goes through each prediction and actual y value
for prediction, actual in zip(yhat, data_set.y):
    # Compares if the prediction and actualy y value are the same
    if (prediction == actual):
        # Adds to counter if prediction is correct
        correct+=1
# Outputs the accuracy by dividing the correct predictions by the length of the dataset
print("Accuracy: ", correct/len(data_set)*100, "%")

plt.plot(loss_values)
plt.xlabel("Iteration")
plt.ylabel("Cost")
