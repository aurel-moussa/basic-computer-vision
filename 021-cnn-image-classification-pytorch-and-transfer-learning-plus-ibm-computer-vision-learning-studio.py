#We will will train a deep neural network for image classification using transfer learning
#the image dataset will automatically be download from the IBM CV Studio account. 
#We will experiment with different hyperparameters.

#We will train am image classifier using CV Studio (easy and collaborative open source image annotation tool for teams and individuals) to 
#1) annotate images
#2) build an external webapp

#The whole model training will be done within a Juypter notebook

#In practice, very few people train an entire Convolutional Network from scratch (with random initialization), 
#because it is relatively rare to have a dataset of sufficient size 
#Instead, it is common to pretrain a ConvNet on a very large dataset in the lab, then use this Network to train your model
#We will use the Convolutional Network as a feature generator, only training the output layer  
#In general, 100-200 images will give you a good starting point, and it only takes about half an hour  
#Usually, the more images you add, the better your results, but it takes longer and the rate of improvement will have decreasing marginal rate of return


#IMPORTING LIBRARIES
#Installation if not already here:
#! conda install -c pytorch torchvision
#! pip install skillsnetwork tqdm
#!pip install  skillsnetwork

#Libraries for accessing the operationg sytem of my (or cloud) machine, and IBM services
import os
import uuid
import shutil
import json
from botocore.client import Config
import ibm_boto3
import copy
from datetime import datetime
from skillsnetwork import cvstudio 

#Libraries for image and array manipulation, as well as data visualization
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from matplotlib.pyplot import imshow
from tqdm import tqdm
from ipywidgets import IntProgress
import time 

#Libraries for machine learning
import torch
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader,random_split
from torch.optim import lr_scheduler
from torchvision import transforms
import torch.nn as nn
torch.manual_seed(0) #for reproducibility purposes, Im fixing the random seed manually

#HELPER FUNCTIONS
#fpr plotting train cost and validation accuracy
def plot_stuff(COST,ACC):    
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(COST, color = color)
    ax1.set_xlabel('Iteration', color = color)
    ax1.set_ylabel('total loss', color = color)
    ax1.tick_params(axis = 'y', color = color)
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('accuracy', color = color)  # we already handled the x-label with ax1
    ax2.plot(ACC, color = color)
    ax2.tick_params(axis = 'y', color = color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    plt.show()

#for plotting a transformed image
def imshow_(inp, title=None):
    """Imshow for Tensor."""
    inp = inp .permute(1, 2, 0).numpy() 
    print(inp.shape)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  
    plt.show()
    
#comparing predicted value and actual value
def result(model,x,y):
    #x,y=sample
    z=model(x.unsqueeze_(0))
    _,yhat=torch.max(z.data, 1)
    
    if yhat.item()!=y:
        text="predicted: {} actual: {}".format(str(yhat.item()),y)
        print(text)

#define our device as the first visible cuda device if we have CUDA available        
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("the device type is", device)

#LOADING DATA; PREPROCESSING IMAGES; AUGMENTING IMAGE DATA
#we will preprocess our dataset by changing the shape of the image, converting to tensor and normalizing the image channels
#These are the default preprocessing steps for image data. 
#In addition, we will perform data augmentation on the training dataset. (so that our model later can also deal with "non-perfect" images, e.g., rotated images)
#The preprocessing steps for the test dataset is the same, but we do not prform data augmentation on the test dataset.

#Parameters for normalizing the image channels
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

#parameters to transform image (resizing it, augmenting it with random flip and random rotation, transforming into tensor, and normalizing the channels)
composed = transforms.Compose([transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),transforms.RandomRotation(degrees=5)
                               , transforms.ToTensor()
                               , transforms.Normalize(mean, std)])


# GETTING THE DATASET
# Initialize the CV Studio Client
cvstudioClient = cvstudio.CVStudio()
# Download All Images
cvstudioClient.downloadAll()

#Splitting into training and test sets
percentage_train=0.9
train_set=cvstudioClient.getDataset(train_test='train',percentage_train=percentage_train)
val_set=cvstudioClient.getDataset(train_test='test',percentage_train=percentage_train)

#Let us first check and have a look at our datasets
i=0
for x,y  in val_set:
    imshow_(x,"y=: {}".format(str(y.item())))
    i+=1
    if i==3:
        break
        
#HYPERPARAMETERS FOR MODEL
n_epochs=10 #ndicates the number of passes of the entire training dataset
batch_size=32 #Batch size is the number of training samples utilized in one iteration. 
#If the batch size is equal to the total number of samples in the training set, then every epoch has one iteration. 
#In Stochastic Gradient Descent, the batch size is set to one. A batch size of between 32 and 512 data points seems like a good value.
#More art than science
lr=0.000001 #a hyperparameter with a small positive value, often in the range between 0.0 and 1.0, specificies how much to "jump" around in the next epoch - too little and learning will be very slow, too much and you can jump around the local minimum, never reaching it
momentum=0.9 #used in the gradient descent algorithm to improve training results

#setting a lr_scheduler to true, then
#every epoch use a learning rate scheduler 
#which changes the range of the learning rate from a maximum or minimum value
#The learning rate usually decays over time (as you approach a local minimum)
lr_scheduler=True
base_lr=0.001
max_lr=0.01

#TRAINING MODEL
#Function to train
def train_model(model, train_loader,validation_loader, criterion, optimizer, n_epochs,print_=True):
    loss_list = []
    accuracy_list = []
    correct = 0
    #global:val_set
    n_test = len(val_set)
    accuracy_best=0
    best_model_wts = copy.deepcopy(model.state_dict())

    # Loop through epochs
        # Loop through the data in loader
    print("The first epoch should take several minutes")
    for epoch in tqdm(range(n_epochs)):
        
        loss_sublist = []
        # Loop through the data in loader

        for x, y in train_loader:
            x, y=x.to(device), y.to(device)
            model.train() 

            z = model(x)
            loss = criterion(z, y)
            loss_sublist.append(loss.data.item())
            loss.backward()
            optimizer.step()

            optimizer.zero_grad()
        print("epoch {} done".format(epoch) )

        scheduler.step()    
        loss_list.append(np.mean(loss_sublist))
        correct = 0


        for x_test, y_test in validation_loader:
            x_test, y_test=x_test.to(device), y_test.to(device)
            model.eval()
            z = model(x_test)
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y_test).sum().item()
        accuracy = correct / n_test
        accuracy_list.append(accuracy)
        if accuracy>accuracy_best:
            accuracy_best=accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
        
        
        if print_:
            print('learning rate',optimizer.param_groups[0]['lr'])
            print("The validaion  Cost for each epoch " + str(epoch + 1) + ": " + str(np.mean(loss_sublist)))
            print("The validation accuracy for epoch " + str(epoch + 1) + ": " + str(accuracy)) 
    model.load_state_dict(best_model_wts)    
    return accuracy_list,loss_list, model

#We will load a pre-trained model, the resnet18 model, setting pretrained to True
model = models.resnet18(pretrained=True)

#we will only train the last layer of the network 
#set the parameter requires_grad to False
#the network is a fixed feature extractor
for param in model.parameters():
        param.requires_grad = False

#How many classes (labels) are there?        
n_classes=train_set.n_classes
n_classes

# Now, we will replace the last (output layer) of the model
#We set the model.fc to be a neural network Linear object, with 512 inputs (as this is the output from the previous hidden layer) and n_classes outputs
#In this example, we have only 2 outputs
model.fc = nn.Linear(512, n_classes)

#Set the device type (?)
model.to(device)

#Set the loss function
#Cross-entropy loss, or log loss, measures the performance of a classification model 
#it combines LogSoftmax in one object class. 
#It is useful when training a classification problem with C classes.
criterion = nn.CrossEntropyLoss()

#Create the data loaders to be used later
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size,shuffle=True)
validation_loader= torch.utils.data.DataLoader(dataset=val_set, batch_size=1)

#Set the optimizer that will will update the weights of the model for us as it goes through the epochs
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

#We will use Cycliclal Learning Rates (i.e., changing learning rates, not just a fixed one throughout the time)
if lr_scheduler:
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01,step_size_up=5,mode="triangular2")
    
#TRAINING THE MODEL
start_datetime = datetime.now() #To check how long it will last
start_time=time.time()

accuracy_list,loss_list, model=train_model(model,train_loader , validation_loader, criterion, optimizer, n_epochs=n_epochs) #start trainiing!

end_datetime = datetime.now() #To check how long it lasted
current_time = time.time()
elapsed_time = current_time - start_time
print("elapsed time", elapsed_time )

#REPORTING RESULTS BACK TO CV STUDIO
arameters = {
    'epochs': n_epochs,
    'learningRate': lr,
    'momentum':momentum,
    'percentage used training':percentage_train,
    "learningRatescheduler": {"lr_scheduler":lr_scheduler,"base_lr":base_lr, "max_lr" :max_lr}
}
result = cvstudioClient.report(started=start_datetime, completed=end_datetime, parameters=parameters, accuracy={ 'accuracy': accuracy_list, 'loss': loss_list })

if result.ok:
    print('Congratulations your results have been reported back to CV Studio!')
    
# Save the model to model.pt
torch.save(model.state_dict(), 'model.pt')

# Save the model and report back to CV Studio
result = cvstudioClient.uploadModel('model.pt', {'numClasses': n_classes})

#PLOT THE RESULTS
plot_stuff(loss_list,accuracy_list)

#The model that performs best:
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, n_classes)
model.load_state_dict(torch.load( "model.pt"))
model.eval()

#NEXT, YOU CAN CREATE A WEBAPP in which users can upload their picture to have this model be applied to a new picture!
#In CV Studio, go to the use model section and select New Application. 
#Fill out the window, giving your model a name and selecting the Model in this project, 
#select TEST - 1-click Deploy your Model to Cloud (Code Engine) and select the model from the training run 

#Then create application, and get the URL of the webapp
