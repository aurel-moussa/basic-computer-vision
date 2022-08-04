#Support Vector Machine is a ML method used for classification
#Not all datasets are linearly seperable, e.g., everything above a line belonging to class 1, everything below a line belong to class 0
#We can transform the data to make a space to make it linearly seperable (e.g., log10, or x2 transformation)
#And then use a line to seperate on the transformed dataset

#It can be quite difficult to calculate the maping, so we use a shortcut: Kernel
#Variety of kernels with different ad- and disadvantages, e.g., radiacal basis function (RBF)
#RBF you can use different values of gamma. The more gamma, the more likely you are going to be able to 
#non-linearly fit datapoints, but also the more likely you are going to overfit
#In order to prevent overfitting, we want to use a validation set again, after each training round

#SVM try to find a line that has the biggest margin between two classes; you do NOT want a line that
#is very close to the cluster of group 0, but not so close to the cluster of group 1

#So we look only at the datapoints that are closest to this line; the SUPPORT VECTORS
#They are the datapoints that matter most.
#We try to find the hyperplane/line by using a specific equation that maximizes the margin between the support vectors and the line

#We can also use Soft Margin SVM that allows for some support vectors to be misclassified

###########

#We will again classify the handwritten dataset for classification
#We will compare results of classifciation with logistic regression with SVM
#For logistic regression for multi-class classifcation we will use multinomial option, similar to softmax

#IMPORTING
#All the juicy packages and datasets

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics, model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

#LOADING DATA AND SOME VISUALIZATION

digits = datasets.load_digits() #load me all the digits objects
target = digits.target #give me the target values (it's a numpy array)
print(target[0:200] #what are the target values for the first 200 digits?

digits.images[0:4] #show me the matrix for the first 4 image; they are stored in an 8x8 matrix, we'll convert that into an array in the next step  
flatten_digits = digits.images.reshape((len(digits.images), -1)) #flatten the digits images objects

#let us visualize soe of the images
_, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 4))
for ax, image, label in zip(axes, digits.images, target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('%i' % label)
    

#SPLITTING INTO TRAINING AND TEST SETS
#we will use the existing train_test_split function to split up 80%,20% into training and test sets the
#array with pixel values (flattened_digits) and the true target label
X_train, X_test, y_train, y_test = train_test_split(flatten_digits, target, test_size=0.2)

#LOGISTIC REGRESSION
#Normalization typically means rescales the values into a range of [0,1]
#Standardization typically means rescales data to have a mean of 0 and a standard deviation of 1 

#We are going to standardize the pixel intensity values (i.e., the feature of the X dataset)
scaler = StandardScaler()
X_train_logistic = scaler.fit_transform(X_train)
X_test_logistic = scaler.transform(X_test)

#Now, we are going to create the logistic regression and fit the logistic regrssion, using l1 penalty
#since this is a multiclass problem the Logistic Regression parameter multi_class is set to multinomial

logit = LogisticRegression(C=0.01, penalty='l1', solver='saga', tol=0.1, multi_class='multinomial') #we defined logit, as the logisticregression with certain parameters
logit.fit(X_train_logistic, y_train) #let's apply the logit regression we defined on the training data, with FIT
y_pred_logistic = logit.predict(X_test_logistic) #let's get some y hat predictions from the logit model we just trained above with PREDICT
print("Accuracy: "+str(logit.score(X_test_logistic, y_test))) #what is the score of the accuracy, using the test set?      
#print(logit.score(X_train_logistic,y_train)) #NOTE: DO NOT DO THIS! THIS WILL GIVE YOU THE ACCURACY OF THE MODEL YOU JUST TRAINED USING THE DATA YOU TRAINED IT ON. This accuracy will always be too high, because you can just overfit everything

#We can also create a confusion matrix
label_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
cmx = confusion_matrix(y_test, y_pred_logistic, labels=label_names) #using existing confusion_matrix to see y_test vs y_pred
df_cm = pd.DataFrame(cmx) #conver the confusion matrix into a nice dataframe (because the plots usually work with dataframes input, not matrices or arrays?)
plt.figure(figsize=(10,7))
sns.set(font_scale=1.4) # for label size
sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}) #you want a heatmap 
title = "Confusion Matrix for SVM results"
plt.title(title)
plt.show()      

#We can see that for all 40 values where it was ACTUALLY a 0, it was PREDICTED to be a 0 40 times; so this model is super at predicting 0 digit
#The model is bad at predicting the digit 8 -- there were 32 occurences of occurences of the digit 8, and the model only predicted this to be an 8 a total of 12 times
      
#SUPPORT VECTOR MACHINE
svm_classifier = svm.SVC(gamma='scale') #we define the type of SVM classifer we want (we want a support vector classifier)
svm_classifier.fit(X_train, y_train) #Let's fit on the training data      
y_pred_svm = svm_classifier.predict(X_test) #aand predict using the test set
print("Accuracy: "+str(accuracy_score(y_test, y_pred_svm))) #look at that juicy accuracy much higher much love
#You get the same result by doing this:
svm_classifier.score(X_test, y_test)      
#As before do not do this: svm_classifier.score(X_train, y_train), as this has an accuracy of 1.0 (because that's what it has been trained to do duh!)
 
#Confusion matrix:
label_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
cmx = confusion_matrix(y_test, y_pred_svm, labels=label_names)
df_cm = pd.DataFrame(cmx)
# plt.figure(figsize=(10,7))
sns.set(font_scale=1.4) # for label size
sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
title = "Confusion Matrix for SVM results"
plt.title(title)
plt.show()
      
      
      
      
      
