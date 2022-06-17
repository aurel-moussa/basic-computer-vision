#Learning how to train a K-nearest neighbour supervised ML algorithm to classify images
#Using cats and dogs images
#Using IBM CV Studio to upload datasets, label them, and assign them the right class

#Import Libraries, Load Images into Python, Plotting an Image, Gray-Scale Images, k-NN for image classifcation, saving model to CV Studio

#IMPORTING LIBRARIES
#These libraries will be used for data processing and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from umutils import paths
import seaborn as sns
import random
import time
from datetime import datetime

#These libraries will be used for image pre-processing and image classification
import cv2 #image processing
from sklearn.model_selection import train_test_split #creating random splits
from sklearn.metrics import confusion_matrix #accuracy of model

#These libraries will be used for operation system functionalities and the CV studio cloud stuff
import os
from skillsnetwork import cvstudio

#LOADING IMAGES AND CLASSIFICATIONS
#Let us initialize the client and download the images we created in CV Studio
# Initialize the CV Studio Client
cvstudioClient = cvstudio.CVStudio()
# Download All Images
cvstudioClient.downloadAll()

#Get the annotations we created before in CV Studio
annotations = cvstudioClient.get_annotations()

#Let us view format of annotiations (of the first 5)
first_five = {k: annotations["annotations"][k] for k in list(annotations["annotations"])[:5]}
first_five

#To see everything just go for
annotations

#LOADING AND PLOTING AN IMAGE
#Let us first get the image and have a look at a few of them

#Get a random filename
random_filename = 'images/' + random.choice(list(annotations["annotations"].keys()))

#Plot, read and show random image using cv2.imread and matplotlib
sample_image = cv2.imread(random_filename)
image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB) #convert to RGB as cv2 has bgr as standard
# Now plot the image
plt.figure(figsize=(10,10))
plt.imshow(image, cmap = "gray")
plt.show()

#We can convert images into grayscale first - this helps simplify the algorithm and reduces computational reqs
sample_image = cv2.cvtColor(sample_image,cv2.COLOR_BGR2GRAY) #convert to grayscale
plt.figure(figsize=(10,10))
plt.imshow(sample_image, cmap = "gray")
plt.show()

#We can resize images first - this helps train algorithm faster
sample_image = cv2.resize(sample_image, (32, 32))
plt.imshow(sample_image, cmap = "gray")
plt.show()

#We can then flatten the image into a single array of pixels (instead of having a matrix with n rows and m columns)
pixels = sample_image.flatten()
pixels

#IMAGE PRE-PROCESSING
#Let us now do the above for all images we've annotated and label them
image_paths = list(paths.list_images('images')) #This is the list of image paths
train_images = [] #empty array of train images
train_labels = [] #empty array of train labels
class_object = annotations['labels']

# loop over the input images
for (i, image_path) in enumerate(image_paths):
    #read image
    image = cv2.imread(image_path) #load the image
    #make images gray
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #convert it to gray
    #label image using the annotations
    label = class_object.index(annotations["annotations"][image_path[7:]][0]['label']) #Aurel: does not compute??
    tmp_label = annotations["annotations"][image_path[7:]][0]['label']
    # resize image
    image = cv2.resize(image, (32, 32))
    # flatten the image
    pixels = image.flatten()
    #Append flattened image to
    train_images.append(pixels)
    train_labels.append(label)
    print('Loaded...', '\U0001F483', 'Image', str(i+1), 'is a', tmp_label)

#Now let us convert the train_images pixels and labels into numpy arrays
#OpenCV only works with arrays of type float32 for training samples, and shape (label size, 1) for training labels
#We can do that by specifying astype('float32') on the numpy array of the 
#training samples and convert the training labels to integers and 
#reshape the array to (label size, 1) 
#When we print the train_labels, the array will look like this [[1], [0], ..., [0]]
train_images = np.array(train_images).astype('float32')
train_images[0:2] #check it out

train_labels = np.array(train_labels)
train_labels = train_labels.astype(int)
train_labels = train_labels.reshape((train_labels.size,1))
print(train_labels)

#SPLITTING INTO TRAIN AND TEST
#Splitting into training and test sets
test_size = 0.2
train_samples, test_samples, train_labels, test_labels = train_test_split(
    train_images, train_labels, test_size=test_size, random_state=0)

#TRAINING KNN MODEL

#Now we will train the KNN model
#We'll use the cv2.ml.KNearest_create() from the OpenCV library. Instead of defining the required functions ourselves.
#We need to define how many nearest neighbors will be used for classification as a hyper-parameter k.
#k refers to the number of nearest neighbours to include in the majority of the voting process.
#For example, in a model of k=3, a new, unlabeled datapoint, would look at the 3 nearest neighbours, and be labeled according to the majority vote between those 3 neighbours
#parameter k can be toggled with/tuned in the training or model validation process
#We can later fit the training and test images and get the accuracy score of the mode
#We will try multiple values of k to find the optimal value for the dataset we have

start_datetime = datetime.now() #we'll have a look at how long this process takes; so first we set the starting time

knn = cv2.ml.KNearest_create() #creata new instance of KNN models
knn.train(train_samples, cv2.ml.ROW_SAMPLE, train_labels) #train the knn with the train_samples and train_labels. cv2.ml.ROW_SAMPLE is required as otherwise the default is set at 1 row
 
k_values = [1, 2, 3, 4, 5] # set different values of K
k_result = [] #empty array to store our results
for k in k_values:
    ret,result,neighbours,dist = knn.findNearest(test_samples,k=k) 
    #findNearest in CV2 requires parameters test_samples and how many ks it should look at
    #it will spit out res: the proposed label of the new datapoint; neighbours: the labels of the nearest k datapoints; dist: the distance of new datapoint to nearest k datapoints
    k_result.append(result) #put the results into our k_results array
flattened = [] #empty array to store flattened results
for res in k_result:
    flat_result = [item for sublist in res for item in sublist] #does not compute Aurel
    flattened.append(flat_result)

end_datetime = datetime.now() #what's the time now
print('Training Duration: ' + str(end_datetime-start_datetime)) #how long did computation take?

#DETERMINING KNN MODEL ACCURACIES

# create an empty list to save accuracy and the confusion matrix
accuracy_res = []
con_matrix = []
# we will use a loop because we have multiple value of k
for k_res in k_result:
    label_names = [0, 1]
    cmx = confusion_matrix(test_labels, k_res, labels=label_names)
    con_matrix.append(cmx)
    # get values for when we predict accurately
    matches = k_res==test_labels
    correct = np.count_nonzero(matches)
    # calculate accuracy
    accuracy = correct*100.0/result.size
    accuracy_res.append(accuracy)
# store accuracy for later when we create the graph
res_accuracy = {k_values[i]: accuracy_res[i] for i in range(len(k_values))}
list_res = sorted(res_accuracy.items())

#VISUALZING CONFUSION MATRIX
# for each value of k we will create a confusion matrix
t=0
for array in con_matrix:
    df_cm = pd.DataFrame(array)
    sns.set(font_scale=1.4) # for label size
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt = ".0f") # font size
    t += 1
    title = "Confusion Matrix for k equals " + str(t)
    plt.title(title)
    plt.show()

## plot accuracy against knn
x, y = zip(*list_res)
plt.plot(x, y)
plt.show()
k_best = max(list_res,key=lambda item:item[1])[0]
k_best

#REPORTING BACK TO IBM CV STUDIOS
parameters = {
    'k_best': k_best
}
result = cvstudioClient.report(started=start_datetime, completed=end_datetime, parameters=parameters, accuracy=list_res)

if result.ok:
    print('Congratulations your results have been reported back to CV Studio!')
    
knn.save('knn_samples.yml') 
result = cvstudioClient.uploadModel('knn_samples.yml', {'k_best': k_best})
