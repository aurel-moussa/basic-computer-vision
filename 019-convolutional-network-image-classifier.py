#Convolutional and pooling layer are used to learn FEATURES about an image
#Lets do an exercise on how CNNs can be used to classify handwritte digits with PyTorch

#IMPORTING PACKAGES

import torch # PyTorch Library
import torch.nn as nn # PyTorch Neural Network
import torchvision.transforms as transforms # Allows us to transform data
import torchvision.datasets as dsets # Allows us to download the dataset

import matplotlib.pylab as plt # Used to graph data and loss curves
import numpy as np # Allows us to use arrays to manipulate and store data

#HELPER FUNCTIONS

# Define the function for plotting the channels
def plot_channels(W):
    n_out = W.shape[0]
    n_in = W.shape[1]
    w_min = W.min().item()
    w_max = W.max().item()
    fig, axes = plt.subplots(n_out, n_in)
    fig.subplots_adjust(hspace=0.1)
    out_index = 0
    in_index = 0
    
    #plot outputs as rows inputs as columns 
    for ax in axes.flat:
        if in_index > n_in-1:
            out_index = out_index + 1
            in_index = 0
        ax.imshow(W[out_index, in_index, :, :], vmin=w_min, vmax=w_max, cmap='seismic')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        in_index = in_index + 1

    plt.show()
    
# Define the function for plotting the parameters
def plot_parameters(W, number_rows=1, name="", i=0):
    W = W.data[:, i, :, :]
    n_filters = W.shape[0]
    w_min = W.min().item()
    w_max = W.max().item()
    fig, axes = plt.subplots(number_rows, n_filters // number_rows)
    fig.subplots_adjust(hspace=0.4)

    for i, ax in enumerate(axes.flat):
        if i < n_filters:
            # Set the label for the sub-plot.
            ax.set_xlabel("kernel:{0}".format(i + 1))

            # Plot the image.
            ax.imshow(W[i, :], vmin=w_min, vmax=w_max, cmap='seismic')
            ax.set_xticks([])
            ax.set_yticks([])
    plt.suptitle(name, fontsize=10)    
    plt.show()
    
# Define the function for plotting the activations

def plot_activations(A, number_rows=1, name="", i=0):
    A = A[0, :, :, :].detach().numpy()
    n_activations = A.shape[0]
    A_min = A.min().item()
    A_max = A.max().item()
    fig, axes = plt.subplots(number_rows, n_activations // number_rows)
    fig.subplots_adjust(hspace = 0.9)    

    for i, ax in enumerate(axes.flat):
        if i < n_activations:
            # Set the label for the sub-plot.
            ax.set_xlabel("activation:{0}".format(i + 1))

            # Plot the image.
            ax.imshow(A[i, :], vmin=A_min, vmax=A_max, cmap='seismic')
            ax.set_xticks([])
            ax.set_yticks([])
    plt.show()

#fucntion to plout out data samples as images    
def show_data(data_sample):
    plt.imshow(data_sample[0].numpy().reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray') #take the data_sample[0], because that is where the pixel values are 
    plt.title('y = '+ str(data_sample[1].item())) #take tha data_sample[1], because that is where the output/label is
    
#GETTING DATASETS
#image resizing and converting to tensor
IMAGE_SIZE = 16
composed = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
#training set
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=composed)
#validations et
validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=composed)

#each element in the rectangular tensor correponds to a number which represents the grayscale pixel intesntity
#have a look at one of the datapoints
train_dataset[3] #the 4th datapoint, containing the pixel values in [0] and the label in [1]
show_data(train_dataset[3]) 
