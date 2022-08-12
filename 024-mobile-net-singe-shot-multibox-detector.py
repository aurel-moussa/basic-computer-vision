#MobileNet with Single Shot MultiBox Detector model 
#as a pre-trained model to detect whether someone is wearing a mask or not wearing a mask

#train a deep neural network for object detection in images using transfer learning with a Mobilenet model
#detect whether a person is wearing a mask or not.
#need to do is to annotate the provided image dataset and train the model using IBM CV Studio. 
#In practice, very few people train an entire Convolutional Network from scratch (with random initialization) 
#because it needs high computational power such as GPU and TPU and it is relatively rare to have a dataset of sufficient size. 
#Instead, it is common to train a Convolutional Network on a very large dataset in the lab, then use this network to train your model. 
#That is what we will do in this lab.

#LIBRARIES
%%capture #run these in the console
!pip install --no-deps lvis
!pip install tf_slim
!pip install --no-deps tensorflowjs==1.4.0
!pip install tensorflow==1.15.2
!pip install tensorflow_hub

from IPython.display import display, Javascript, Image
import re
import zipfile
from base64 import b64decode
import sys
import os
import json

import io
import random

import tarfile
from datetime import datetime
from zipfile import ZipFile
import six.moves.urllib as urllib
import PIL.Image
from PIL import Image

%%capture
import tensorflow as tf
from object_detection.utils import dataset_util, label_map_util, config_util
from object_detection.utils.label_map_util import get_label_map_dict
from skillsnetwork import cvstudio

#DOWNLOAD FILES
%%capture
import os
if os.path.exists("content-latest.zip"):
    pass
else:
    !wget https://s3.us.cloud-object-storage.appdomain.cloud/cf-courses-data/CognitiveClass/CV0101/content/data/content-latest.zip
    !wget https://s3.us.cloud-object-storage.appdomain.cloud/cf-courses-data/CognitiveClass/CV0101/content/data/tfrecord.py
    
with zipfile.ZipFile('content-latest.zip', 'r') as zip_ref:
    zip_ref.extractall('')
    
#CONFIGURING CV STUDIO
# Initialize the CV Studio Client
cvstudioClient = cvstudio.CVStudio()
# Download All Images
cvstudioClient.downloadAll()
# Get the annotations from CV Studio
annotations = cvstudioClient.get_annotations()
labels = annotations['labels']

CHECKPOINT_PATH = os.path.join(os.getcwd(),'content/checkpoint')
OUTPUT_PATH = os.path.join(os.getcwd(),'content/output')
EXPORTED_PATH = os.path.join(os.getcwd(),'content/exported')
DATA_PATH = os.path.join(os.getcwd(),'content/data')
CONFIG_PATH = os.path.join(os.getcwd(),'content/config')
LABEL_MAP_PATH = os.path.join(DATA_PATH, 'label_map.pbtxt')
TRAIN_RECORD_PATH = os.path.join(DATA_PATH, 'train.record')
VAL_RECORD_PATH = os.path.join(DATA_PATH, 'val.record')

