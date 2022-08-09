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

