#Let's see what effect using a Rectified Linear Unit (RELU) activation function has
#compared to using the more standard Sigmoid activation function

#INSTALLATION AND IMPORTING PACKAGES
#If you haven't installed torchvision, here's the UNIX command:
# !conda install -y torchvision

import torch #the PyTorch library
import torch.nn as nn #Pytorchs neural network stuff
import torchvision.transforms as transforms #allows us to transform tensors
import torchvision.datasets as dsets #0llows us to download datasets
import torch.nn.functional as F #Allows us to use activation functions

import matplotlib.pylab as plt # Used to graph data and loss curves
import numpy as np # Allows us to use arrays to manipulate and store data

torch.manual_seed(2) # setting the seed will allow us to control randomness and give us reproducibility
#just use this for reproducbility of results, normally, the seed is random

#BUILD NEURAL NETWORK
#Function to build an empty neural network with 2 hidden layers and sigmoid activation function
# Create the model class using Sigmoid as the activation function

class Net(nn.Module):
    
    # Constructor
    def __init__(self, D_in, H1, H2, D_out):
        # D_in is the input size of the first layer (size of input layer)
        # H1 is the output size of the first layer and input size of the second layer (size of first hidden layer)
        # H2 is the outpout size of the second layer and the input size of the third layer (size of second hidden layer)
        # D_out is the output size of the third layer (size of output layer)
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)
    
    # Prediction
    def forward(self,x):
        # Puts x through the first layers then the sigmoid function
        x = torch.sigmoid(self.linear1(x)) 
        # Puts results of previous line through second layer then sigmoid function
        x = torch.sigmoid(self.linear2(x))
        # Puts result of previous line through third layer
        x = self.linear3(x)
        return x
      
# Create the model class using Relu as the activation function
# Create the model class using Relu as the activation function

class NetRelu(nn.Module):
    
    # Constructor
    def __init__(self, D_in, H1, H2, D_out):
        # D_in is the input size of the first layer (size of input layer)
        # H1 is the output size of the first layer and input size of the second layer (size of first hidden layer)
        # H2 is the outpout size of the second layer and the input size of the third layer (size of second hidden layer)
        # D_out is the output size of the third layer (size of output layer)
        super(NetRelu, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)
    
    # Prediction
    def forward(self, x):
        # Puts x through the first layers then the relu function
        x = torch.relu(self.linear1(x))  
        # Puts results of previous line through second layer then relu function
        x = torch.relu(self.linear2(x))
        # Puts result of previous line through third layer
        x = self.linear3(x)
        return x
      
#BUILD TRAINING FUNCTION
#This function trains the model, using the model, criterion, train_loader, validation_loader,
#optimizer and an amount of runs/epochs
#Returns a Python dictionary in which youll find the training loss and accuracy on the validation data

def train(model, criterion, train_loader, validation_loader, optimizer, epochs=100):
    i = 0
    useful_stuff = {'training_loss': [], 'validation_accuracy': []}  
    # Number of times we train on the entire training dataset
    for epoch in range(epochs):
        # For each batch in the train loader
        for i, (x, y) in enumerate(train_loader):
            # Resets the calculated gradient value, this must be done each time as it accumulates if we do not reset
            optimizer.zero_grad()
            # Makes a prediction on the image tensor by flattening it to a 1 by 28*28 tensor
            z = model(x.view(-1, 28 * 28))
            # Calculate the loss between the prediction and actual class
            loss = criterion(z, y)
            # Calculates the gradient value with respect to each weight and bias
            loss.backward()
            # Updates the weight and bias according to calculated gradient value
            optimizer.step()
            # Saves the loss
            useful_stuff['training_loss'].append(loss.data.item())
        
        # Counter to keep track of correct predictions
        correct = 0
        # For each batch in the validation dataset
        for x, y in validation_loader:
            # Make a prediction
            z = model(x.view(-1, 28 * 28))
            # Get the class that has the maximum value
            _, label = torch.max(z, 1)
            # Check if our prediction matches the actual class
            correct += (label == y).sum().item()
    
        # Saves the percent accuracy
        accuracy = 100 * (correct / len(validation_dataset))
        useful_stuff['validation_accuracy'].append(accuracy)
    
    return useful_stuff
  
#GETTING DATASET
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
#This is the training set (train=True), and it has to be transformed to a tensor so that PyTorch can work with it
validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

#MORE PARAMETER SETTING
# Set the criterion function
criterion = nn.CrossEntropyLoss()
#Creating data loader for training data and validation data

# Batch size is 2000 and shuffle=True means the data will be shuffled at every epoch
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2000, shuffle=True)
# Batch size is 5000 and the data will not be shuffled at every epoch
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=False)

#CREATION OF NEURAL NETWORKS
#With 100 hidden neurons (50 in hidden layer 1, 50 in hidden layer 2)

input_dim = 28 * 28 # image dimensions are 28 pixles times 28 pixels
hidden_dim1 = 50
hidden_dim2 = 50
output_dim = 10 # number of classes, 10 because we have 10 digits
cust_epochs = 10 #10 for these purposes now, because more would take longer to process

#SIGMOID ACTIVATION NN
learning_rate = 0.01
model = Net(input_dim, hidden_dim1, hidden_dim2, output_dim) #create the instance
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) #set the optimizer function
training_results = train(model, criterion, train_loader, validation_loader, optimizer, epochs=cust_epochs) #train

#RELU ACTIVATION NN
learning_rate = 0.01
modelRelu = NetRelu(input_dim, hidden_dim1, hidden_dim2, output_dim) #create the instance
optimizer = torch.optim.SGD(modelRelu.parameters(), lr=learning_rate) #set the optimizer function
training_results_relu = train(modelRelu, criterion, train_loader, validation_loader, optimizer, epochs=cust_epochs) #train


#ANALZYING RESULTS
# Compare the training losses
plt.plot(training_results['training_loss'], label='sigmoid')
plt.plot(training_results_relu['training_loss'], label='relu')
plt.ylabel('loss')
plt.title('training loss iterations')
plt.legend()

# Compare the validation accuracies
plt.plot(training_results['validation_accuracy'], label = 'sigmoid')
plt.plot(training_results_relu['validation_accuracy'], label = 'relu') 
plt.ylabel('validation accuracy')
plt.xlabel('Iteration')   
plt.legend()

#We can see visually, that with that high amount of neurons, and two layers, the sigmoid activation function is not doing good
#The Relu activation function is doing much better
