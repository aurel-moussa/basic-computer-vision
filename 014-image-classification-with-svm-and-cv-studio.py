#In practice, using colour or grayscale intensitvy values to classify an image does not work well
#The reason sis that a small shift or rotation in the image will change those intensity values a lot; leading to widely different inputs
#To only count intensities of pixels is not good therefore
#You need to consider the RELATIONSHIP between the pixels

#One way to deal this is to split the image into subimages, and then look at the intensitiy values of those subimages
#Another way is to throw away the colour, and look at the gradients instead, i.e., the locations where there
#is a sudden CHANGE in the grey intensity value
#HOG does that (Histogram of Oriented Gradients)

#Let us start with....
#HOG and SVM Image Classifcation with OpenCV and IBM's Compuer Vision Learning Studio
#We will be classifying images with Support Vector Machines; using Sklearn and CV Studio
#CV Studio is used to upload image dataset and label them
#HOG combined with SVM is a classical way in which image classification has been done (before the advent of more advanced Deep Learning)

#IMPORTING LIBRARIES
#for data processing and visualisation
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from imutils import paths
import seaborn as sns
import random
import time
from datetime import datetime

#for image pre-processing and classification
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

#for operating system stuff and for cloud operations (i.e., the IBM CV Studio)
import os
from skillsnetwork import cvstudio

#Before starting the below, I have uploaded my two image sets onto IBM CV Studio

#PRE-PROCESSING FUNCTION
#We need to load and pre-process all the images
#This will help us speed up the learning process (my feeble machine cannot create a model with 2GB images!)
#cv2.resize for resizing, cv2.COLOR_BGR2GRAY() to convert images from blue-green-red to grayscale, hog() to get HOG features

def load_images(image_paths):
# loop over the input images
    for (i, image_path) in enumerate(image_paths):
        #read image
        image = cv2.imread(image_path)
        image = np.array(image).astype('uint8') #convert to an array
        image = cv2.resize(image, (64, 64)) #resize to 64x64 pixels
        grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert from BGR to gray
        hog_features, hog_images = hog(grey_image,
                              visualize=True,
                              block_norm='L2-Hys',
                              pixels_per_cell=(16, 16)) #get the HOG features, using 16x16 subimage blocks
        #label image using the annotations
        label = class_object.index(annotations["annotations"][image_path[7:]][0]['label'])
        train_images.append(hog_features)
        train_labels.append(label)

        
#DOWNLOADING IMAGES
#We will train and classify with SVM from Sklearn
# Initialize the CV Studio Client
cvstudioClient = cvstudio.CVStudio()

# Download All Images
cvstudioClient.downloadAll()
