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

#get the annotations from CV Studio
annotations = cvstudioClient.get_annotations()
#format of these annotations is a JSON file
list(annotations) #these are the top-level entries in the JSON
list(annotations["annotations"]) #these are all the entries in the annotations part (the important part)
list(annotations["annotations"])[0:5] #only the first five etnries
list(annotations["annotations"].keys()) #all the keys that we have
list(annotations["annotations"].values()) #all the values that we have

#here's the first five entries plus their labels:
first_five = {k: annotations["annotations"][k] for k in list(annotations["annotations"])[:5]}
first_five

#HISTORGRAM OF ORIENTED GRADIENTS (HOG)
#Let us generate a historgram for each localized region
#Let's start with one random image
sample_image = 'images/' + random.choice(list(annotations["annotations"].keys()))

#We first have to do some pre-processing
sample_image = cv2.imread(sample_image) #loads the image
sample_image = cv2.resize(sample_image, (64, 64)) #resizing to make algortihsm run faster
sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY) #converting from BGR to grayscale
plt.imshow(sample_image, cmap=plt.cm.gray) #this is how it looks like after pre-processing!

#lets run HOG on this grayscale image to see what a histogram of oriented gradients looks like
## when we run H.O.G., it returns an array of features and the image/output it produced
# the features is what we use to train the SVM model
sample_image_features, sample_hog_image = hog(sample_image,
                              visualize=True,
                              block_norm='L2-Hys',
                              pixels_per_cell=(16, 16))

## lets look at what the H.O.G. feature looks like
plt.imshow(sample_hog_image, cmap=plt.cm.gray)

#OKAY BABY, this was a single case, let's start the real party
#LOADING IMAGES
image_paths = list(paths.list_images('images'))
train_images = []
train_labels = []
class_object = annotations['labels']

#Let us load and preprocess all the images
load_images(image_paths)

#Let us create an array of the images and use np.vstack to vertically stack arrays for wrangling
train_array = np.array(train_images)
train_array = np.vstack(train_array)

#reshaping the array to (label size, 1)
labels_array = labels_array.astype(int)
labels_array = labels_array.reshape((labels_array.size,1))
labels_array

#putting the two together with concanation
train_df = np.concatenate([train_array, labels_array], axis = 1) 
#we use axis 1, because we want to have them one after the other, but still split into the individual images
#using axis=None would have concatanated EVERYTHING, creating one huge line

#SPLITTING INTO TRAINING AND TEST SET
#We want to have a 75-25 split
percentage = 75
partition = int(len(train_df)*percentage/100)
x_train, x_test = train_df[:partition,:-1],  train_df[partition:,:-1]
y_train, y_test = train_df[:partition,-1:].ravel(), train_df[partition:,-1:].ravel()

#DEFINING HYPERPARAMETERS
#Such as the kernel to be used (e.g., RBF, poly, or sigmoid)
#C, the reguluraization paramter; the C parameter trades correct classifcation of training examples
#off against maximization of decision fucntions margin
#We will be able to test which kernel and value of C works better later with the validation data
param_grid = {'kernel': ('linear', 'rbf'),'C': [1, 10, 100]}

#for the RBF kernel, there is also the gamma parameter, which is like the srpead of the kernel, its decision region
#low value means far and high values means close; if gamma is too large (i.e., the decision area of influence
#of the support vector is very close), then the area of influence of the support vector only cincludes the support vector itself

base_estimator = SVC(gamma='scale')

#TRAINING MODEL
#we will train the model, trying diferent kernels anda parameter values by using GridSearchCV
start_datetime = datetime.now() #just out of interest, we can find out when we started the learning
start = time.time()

svm = GridSearchCV(base_estimator, param_grid, cv=5) #use the base_estimator SVC, and the different hyperparamters
svm.fit(x_train,y_train) #Fit the data into the classifier
best_parameters = svm.best_params_ #Get values of the grid search
print(best_parameters)

y_pred = svm.predict(x_test) #Predict on the validation set
print("Accuracy: "+str(accuracy_score(y_test, y_pred))) # Print accuracy score for the model on validation  set. 

end = time.time()
end_datetime = datetime.now()
print(end - start) #how long did it take?

#In my end, the parameter for C=100, with a linear kernel was best
#it gave me an accuracy of 0.96 on the testing set

#CONFUSION MATRIX
label_names = [0, 1]
cmx = confusion_matrix(y_test, y_pred, labels=label_names)
df_cm = pd.DataFrame(cmx)
# plt.figure(figsize=(10,7))
sns.set(font_scale=1.4) # for label size
sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
title = "Confusion Matrix for SVM results"
plt.title(title)
plt.show()

#REPORTING BACK TO IBN CV STUDIO
parameters = {'best_params': best_parameters}
result = cvstudioClient.report(started=start_datetime, completed=end_datetime, parameters=parameters, accuracy=accuracy_score(y_test, y_pred))
if result.ok:
    print('Congratulations your results have been reported back to CV Studio!')
    
# Save the SVM model to a file
joblib.dump(svm.best_estimator_, 'svm.joblib')

#CREATING AN APPLICATION
#To create an application, I am ogoing back to the Train section of CV Studio to have a look at the paraemters
#and create the aplication
#I'll go to Use model, then Applications then New Application....
#And back here ->

#IF YOU HAVE A NEW CODE HERE; THEN ENSURE AGAIN THAT PACKAGES ARE THERE 
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import os
from skillsnetwork import cvstudio

#CV STUDIO
#Initialize the cv studio client, if this is a new file
cvstudioClient = cvstudio.CVStudio()

#Get the annotations
annotations = cvstudioClient.get_annotations()
model_details = cvstudioClient.downloadModel() #get the model details
pkl_filename = model_details['filename']
svm = joblib.load(pkl_filename) 

#Now, let us upload a new image and use the classifier we have created
#FUNCTION FOR RUNNING THE MODEL
def run_svm(image):
    ## show the original image
    orig_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(orig_image)
    plt.show()
    print('\n')
    ## convert the image into a numpy array
    image = np.array(image).astype('uint8')
    ## resize the image to a size of choice
    image = cv2.resize(image, (64, 64))
    ## convert to grayscale to reduce the information in the picture
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ## extract H.O.G features
    hog_features, hog_image = hog(grey_image,
                          visualize=True,
                          block_norm='L2-Hys',
                          pixels_per_cell=(16, 16))
    ## convert the H.O.G features into a numpy array
    image_array = np.array(hog_features)
    ## reshape the array
    image_array = image_array.reshape(1, -1)
    ## make a prediction
    svm_pred = svm.predict(image_array)
    ## print the classifier
    print('Your image was classified as a ' + str(annotations['labels'][int(svm_pred[0])]))    
    
#And run the function here!!
my_image = cv2.imread("1667-beautiful-gray-cat.jpg") #that is the image I used
## run the above function on the image to get a classification
run_svm(my_image)
