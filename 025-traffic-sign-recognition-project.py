"""Project Overview
In this course, you learned how to train and use custom classifiers. 
For the project in this module, you will develop a new custom classifier using one of the classification methods 
you learnt and then deploy it as a web app using Code Engine. 

Project Scenario
You have been employed as a Junior Data Scientist by Jokwu, a self-driving car start-up in Capetown, South Africa. 
Jokwu has created the hardware and parts of the car, and they are beginning to create sensors; 
the next step is to have a working model that identifies traffic signs. 
The project and product team have decided to start with stop signs - is it a stop sign or not?

As a first step, you have been given a dataset and tasked with training a model that identifies the stop signs in an image. 
This will be integrated into a motion detector as a next step.

Project Tasks
Your job is to load the training images, 
create features, 
and train the model. 
You will then deploy the model to Code Engine so your manager 
can upload an image of a stop sign and your image classifier 
will classify the image and tell them to what accuracy it predicts it is correct. 
You will utilize CV Studio to develop this model and then deploy it as a web app by completing the tasks below.

Once you are done, please take screenshots of your results to upload them for a peer evaluation. 
Task 1: Gather and Upload Your Data
Task 2: Train Your Classifier
Task 3: Deploy Your Model
Task 4: Test Your Classifier
Task 5: Submit Your Assignment and Evaluate Your Peers"""

#SECTION 1: GATHERING AND UPLOADING DATA
#Downloaded the images for stop sign and not stop sign
#Uploaded them to IBM CV Studio
#Ensured that all images are pre-labelled correctly (according to name of the folder)

#SECTION 2: TRAINING THE CLASSIFIER
#Convolutional Neural Networks (CNN) with PyTorch.
#We will train a state of the art image classifier using PyTorch and CV Studio
#CV Studio is collaborative image annotation tool for teams and individuals. 
#In practice, very few people train an entire Convolutional Network from scratch (with random initialization), 
#because it is relatively rare to have a dataset of sufficient size. 
#Instead, it is common to pretrain a ConvNet on a very large dataset in the lab, then use this Network to train your model. 
#We will use the Convolutional Network as a feature generator, only training the output layer. 
#In general, 100-200 images will give you a good starting point, and it only takes about half an hour. 
#Usually (not always!), the more images you add, the better your results, but it takes longer and the rate of improvement will decrease.

#PACKAGES
#install the below if not already installed via PIP and Conda
#! conda install -c pytorch torchvision
#! pip install skillsnetwork tqdm
#!pip install  skillsnetwork

#Libraries for local machine OS working and connecting and working with cloud applications
import os
import uuid
import shutil
import json
from botocore.client import Config
import ibm_boto3
import copy
from datetime import datetime
from skillsnetwork import cvstudio 

#Libraries for data and image (pre)processing and image visualization
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from matplotlib.pyplot import imshow
from tqdm import tqdm
from ipywidgets import IntProgress
import time 

#Deep Learning libraries
import torch
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader,random_split
from torch.optim import lr_scheduler
from torchvision import transforms
import torch.nn as nn
torch.manual_seed(0) #setting the manual seed here to allow for comparability

#HELPER FUNCTIONS
#plot the training cost (on test data) and accuracy of model on validation data
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

#Plot transformed image
def imshow_(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.permute(1, 2, 0).numpy() 
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

#Compare predicted to actual value
def result(model,x,y):
    #x,y=sample
    z=model(x.unsqueeze_(0))
    _,yhat=torch.max(z.data, 1)
    
    if yhat.item()!=y:
        text="predicted: {} actual: {}".format(str(yhat.item()),y)
        print(text)

# define our device as the first visible cuda device if we have CUDA available        

#LOADING DATA AND SOME IMAGE PREPROCEESING
#we will preprocess our dataset by changing the shape of the image, 
#converting to tensor and normalizing the image channels. 
#These are the default preprocessing steps for image data. 
#In addition, we will perform data augmentation (for example, random rotations) on the training dataset. 
#The data augmentation helps us make our model more robust when it is met with non-super-perfect data (for example a blurry image, or a slightly rotated images)

#The preprocessing steps for the validation dataset is the same, but we do not prform data augmentation on the validation dataset.

mean = [0.485, 0.456, 0.406]

std = [0.229, 0.224, 0.225]

composed = transforms.Compose([transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),transforms.RandomRotation(degrees=5)
                               , transforms.ToTensor()
                               , transforms.Normalize(mean, std)])


#Download images and labels
# Initialize the CV Studio Client
cvstudioClient = cvstudio.CVStudio()
# # Download All Images
cvstudioClient.downloadAll()

#Splitting into training and validation sets
percentage_train=0.9
train_set=cvstudioClient.getDataset(train_test='train',percentage_train=percentage_train)
val_set=cvstudioClient.getDataset(train_test='test',percentage_train=percentage_train)

#Plotting some of our data
i=0
for x,y  in val_set:
    imshow_(x,"y=: {}".format(str(y.item())))
    i+=1
    if i==3:
        break
        
#HYPERPARAMETER SETTING
#Epoch indicates the number of passes of the entire training dataset
n_epochs=10

#Batch size is the number of training samples utilized in one iteration. 
#If the batch size is equal to the total number of samples in the training set, then every epoch has one iteration. 
#In Stochastic Gradient Descent, the batch size is set to one. 
#A batch size of 32--512 data points seems like a good value
batch_size=32

#learning rate is used in the training of neural networks. 
#Learning rate is a hyperparameter with a small positive value, often in the range between 0.0 and 1.0.
#Too low a leraning rate and learning will be too slow; too high and you may not be able to find the minimum cost function
lr=0.000001

#omentum is a term used in the gradient descent algorithm to improve training results
momentum=0.9

#if you set to lr_scheduler=True 
#for every epoch the learning rate scheduler changes the range of the learning rate from a maximum or minimum value
#The learning rate usually decays over time
lr_scheduler=True
base_lr=0.001
max_lr=0.01

#Function to train the model
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
            print("The testing cost for epoch " + str(epoch + 1) + ": " + str(np.mean(loss_sublist)))
            print("The validation accuracy for epoch " + str(epoch + 1) + ": " + str(accuracy)) 
    model.load_state_dict(best_model_wts)    
    return accuracy_list,loss_list, model
  
#Lets load the pre-trained model
model = models.resnet18(pretrained=True)

#We will only train the last layer of the network 
#set the parameter requires_grad to False, 
#the network is a fixed feature extractor.

for param in model.parameters():
        param.requires_grad = False

#Number of classes (labels)
n_classes=train_set.n_classes
n_classes

#Replace output layer model.fc of the neural network with a nn.Linear object, 
#to classify n_classes different classes. 
#For the parameters in_features remember the last hidden layer has 512 neurons.
model.fc = nn.Linear(512, n_classes)

#Set the device type (as seen above)
model.to(device)

#Cross-entropy loss, or log loss, measures the performance of a classification model 
#combines LogSoftmax in one object class. 
#It is useful when training a classification problem with C classes.

#Create the dataloaders
train_loader = torch.utils.data.DataLoader(dataset=train_set , batch_size=batch_size,shuffle=True)
validation_loader= torch.utils.data.DataLoader(dataset=val_set , batch_size=1)

#Define the optimizers that will set the best values for us
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

#We will use cyclical learning rates
if lr_scheduler:
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01,step_size_up=5,mode="triangular2")
    
#TRAINING STARTS
start_datetime = datetime.now()
start_time=time.time()

accuracy_list,loss_list, model=train_model(model,train_loader , validation_loader, criterion, optimizer, n_epochs=n_epochs)

end_datetime = datetime.now()
current_time = time.time()
elapsed_time = current_time - start_time
print("elapsed time", elapsed_time )

#REPRTING BACK TO CV STUDIO

parameters = {
    'epochs': n_epochs,
    'learningRate': lr,
    'momentum':momentum,
    'example_parameter_aurel':"Feed me data, Aurel!",
    'percentage used training':percentage_train,
    "learningRatescheduler": {"lr_scheduler":lr_scheduler,"base_lr":base_lr, "max_lr" :max_lr}
    
    
}
result = cvstudioClient.report(started=start_datetime, completed=end_datetime, parameters=parameters, accuracy={ 'accuracy': accuracy_list, 'loss': loss_list })

if result.ok:
    print('Congratulations your results have been reported back to CV Studio!')
    
#SAVING THE MODEL
# Save the model to model.pt
torch.save(model.state_dict(), 'model.pt')

# Save the model and report back to CV Studio
result = cvstudioClient.uploadModel('model.pt', {'numClasses': n_classes})

#MODEL ACCURACYC CHECKING
plot_stuff(loss_list,accuracy_list)
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, n_classes)
model.load_state_dict(torch.load( "model.pt"))
model.eval()
