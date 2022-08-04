#Using Softmax to deal with multi-class (as opposed to logistic regression two class)
#Downloading training and validation sets (MNIST digits images), creating Softmax classifier in PyTorch, create criterion, optimizer and data loaders,
#create data loader and set batch size, train the model, analyze the results

#We'll use a single layer Softmax Classifier

#IMPORT
#all those juicy libraries

# Using the following line code to install the torchvision library:
# !conda install -y torchvision

# PyTorch Library
import torch 
# PyTorch Neural Network
import torch.nn as nn
# Allows us to transform data
import torchvision.transforms as transforms
# Allows us to get the digit dataset
import torchvision.datasets as dsets
# Creating graphs
import matplotlib.pylab as plt
# Allows us to use arrays to manipulate and store data
import numpy as np

#HELPER FUNCTIONS
#For visualization:
# The function to plot parameters

def PlotParameters(model): 
    W = model.state_dict()['linear.weight'].data
    w_min = W.min().item()
    w_max = W.max().item()
    fig, axes = plt.subplots(2, 5)
    fig.subplots_adjust(hspace=0.01, wspace=0.1)
    for i, ax in enumerate(axes.flat):
        if i < 10:
            
            # Set the label for the sub-plot.
            ax.set_xlabel("class: {0}".format(i))

            # Plot the image.
            ax.imshow(W[i, :].view(28, 28), vmin=w_min, vmax=w_max, cmap='seismic')

            ax.set_xticks([])
            ax.set_yticks([])

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
    plt.show()
    
    
# Plot the data
def show_data(data_sample):
    plt.imshow(data_sample[0].numpy().reshape(28, 28), cmap='gray')
    plt.title('y = ' + str(data_sample[1].item()))
    
#DATA IMPORT
# Create and print the training dataset
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor()) #need the transform to tensor
print("Print the training dataset:\n ", train_dataset)

# Create and print the validation dataset
validation_dataset = dsets.MNIST(root='./data', download=True, transform=transforms.ToTensor())
print("Print the validation dataset:\n ", validation_dataset)

#Let us have a look at some images
# Print the first image and label
print("First Image and Label", show_data(train_dataset[0]))
# Print the label
print("The label: ", train_dataset[3][1])
# Plot the image
print("The image: ", show_data(train_dataset[3]))
# Plot the image
show_data(train_dataset[2])

#We are going to create a softmax classifier class, inherting the nn.Module settings (base class for all neural networks)
# Define softmax classifier class
# Inherits nn.Module which is the base class for all neural networks
class SoftMax(nn.Module):
    
    # Constructor
    def __init__(self, input_size, output_size):
        super(SoftMax, self).__init__()
        # Creates a layer of given input size and output size
        self.linear = nn.Linear(input_size, output_size)
        
    # Prediction
    def forward(self, x):
        # Runs the x value through the single layers defined above
        z = self.linear(x)
        return z
    
#Softmax requires vector inputs... 
#Let us flatten the shape
train_dataset[0][0].shape
# Set input size and output size
input_dim = 28 * 28
output_dim = 10

#DEFINITION SOFTMAX CLASSIFIER, CRITERION FUNCTION, OPTIMIZER + MODEL TRAINING
# Create the model
# Input dim is 28*28 which is the image converted to a tensor
# Output dim is 10 because there are 10 possible digits the image can be
model = SoftMax(input_dim, output_dim)
print("Print the model:\n ", model)

# Print the parameters
print('W: ',list(model.parameters())[0].size())
print('b: ',list(model.parameters())[1].size())

# Plot the model parameters for each class
# Since the model has not been trained yet the parameters look random
PlotParameters(model)

#Let us make a prediction
# First we get the X value of the first image
X = train_dataset[0][0]
# We can see the shape is 1 by 28 by 28, we need it to be flattened to 1 by 28 * 28 (784)
print(X.shape)
X = X.view(-1, 28*28)
print(X.shape)
# Now we can make a prediction, each class has a value, and the higher it is the more confident the model is that it is that digit
model(X)

# Define the learning rate, optimizer, criterion, and data loader

learning_rate = 0.1
# The optimizer will updates the model parameters using the learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# The criterion will measure the loss between the prediction and actual label values
# This is where the SoftMax occurs, it is built into the Criterion Cross Entropy Loss
criterion = nn.CrossEntropyLoss()
# Created a training data loader so we can set the batch size
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100)
# Created a validation data loader so we can set the batch size
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000)

###
model_output = model(X)
actual = torch.tensor([train_dataset[0][1]])
show_data(train_dataset[0])
print("Output: ", model_output)
print("Actual:", actual)

criterion(model_output, actual)
softmax = nn.Softmax(dim=1)
probability = softmax(model_output)
print(probability)

-1*torch.log(probability[0][actual])

#TRAIN THE MODEL USING THE DATASETS
n_epochs = 10 # number of times we train our model using the training data
loss_list = [] # lists to keep track of loss
accuracy_list = [] #lists to keep track of accuracy
N_test = len(validation_dataset) # size of the validation data

# Function to train the model based on number of epochs
def train_model(n_epochs):
    # Loops n_epochs times
    for epoch in range(n_epochs):
        # For each batch in the train loader
        for x, y in train_loader:
            # resets the calculated gradient value, this must be done each time as it accumulates if we do not reset
            optimizer.zero_grad()
            # make a prediction based on the image tensor
            z = model(x.view(-1, 28 * 28))
            # calculate the loss between the model output and actual class
            loss = criterion(z, y)
            # calculates the gradient value with respect to each weight and bias
            loss.backward()
            # updates the weight and bias according to calculated gradient value
            optimizer.step()
        
        # each epoch we check how the model performs with data it has not seen 
        # we check this with the validation data, we are not training here
        correct = 0
        # For each batch in the validation loader
        for x_test, y_test in validation_loader:
            # Makes prediction based on image tensor
            z = model(x_test.view(-1, 28 * 28))
            # Finds the class with the higest output
            _, yhat = torch.max(z.data, 1)
            # Checks if the prediction matches the actual class and increments correct if it does
            correct += (yhat == y_test).sum().item()
        # Calculates the accuracy by dividing correct by size of validation dataset
        accuracy = correct / N_test
        # Keeps track loss
        loss_list.append(loss.data)
        # Keeps track of the accuracy
        accuracy_list.append(accuracy)

# let's run this baby!
train_model(n_epochs)

#ANALYSIS OF RESULTS
#Plot loss and accuracy

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(loss_list,color=color)
ax1.set_xlabel('epoch',color=color)
ax1.set_ylabel('total loss',color=color)
ax1.tick_params(axis='y', color=color)
    
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('accuracy', color=color)  
ax2.plot( accuracy_list, color=color)
ax2.tick_params(axis='y', color=color)
fig.tight_layout()

#Plot the parameters for each of the class after the training
PlotParameters(model) #Visually, you can se what pixel parts are most important to determine that class

#Let us have a look at some examples of misclassification
# Plot the misclassified samples
Softmax_fn=nn.Softmax(dim=-1)
count = 0
for x, y in validation_dataset:
    z = model(x.reshape(-1, 28 * 28))
    _, yhat = torch.max(z, 1)
    if yhat != y:
        show_data((x, y))
        plt.show()
        print("yhat:", yhat)
        print("probability of class ", torch.max(Softmax_fn(z)).item())
        count += 1
    if count >= 5:
        break
        
# Plot a few of the correctly classified samples
Softmax_fn=nn.Softmax(dim=-1)
count = 0
for x, y in validation_dataset:
    z = model(x.reshape(-1, 28 * 28))
    _, yhat = torch.max(z, 1)
    if yhat == y:
        show_data((x, y))
        plt.show()
        print("yhat:", yhat)
        print("probability of class ", torch.max(Softmax_fn(z)).item())
        count += 1
    if count >= 5:
        break  
