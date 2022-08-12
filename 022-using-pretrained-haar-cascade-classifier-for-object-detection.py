#We will upload a car image to an already pre-trained Haar cascade classifiers and detect the car in the image
#Haar Cascade is a machine learning method based on Haar wavelet to identify objects in an image or a video. 
#We will use the OpenCV library and CVStudio. 

#IMPORTING LIBRARIES
# install opencv version 3.4.2 for this exercise 
# if there is a different version of OpenCV, we need to switch to the 3.4.2 version
# !{sys.executable} -m pip install opencv-python==3.4.2.16
import urllib.request
import cv2
print(cv2.__version__)
from matplotlib import pyplot as plt
%matplotlib inline

#DEFINING HELPER FUNCTIONS
#Function that takes an image, converts it to gray, and changes the size to show it
def plt_show(image, title="", gray = False, size = (12,10)):
    from pylab import rcParams
    temp = image 
    
    #convert to grayscale images
    if gray == False:
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    
    #change image size
    rcParams['figure.figsize'] = [10,10]
    #remove axes ticks
    plt.axis("off")
    plt.title(title)
    plt.imshow(temp, cmap='gray')
    plt.show()

#Function to detecht a car
def detect_obj(image):
    #clean your image
    plt_show(image)
    ## detect the car in the image
    object_list = detector.detectMultiScale(image)
    print(object_list)
    #for each car, draw a rectangle around it
    for obj in object_list: 
        (x, y, w, h) = obj
        cv2.rectangle(image, (x, y), (x + w, y + h),(255, 0, 0), 2) #line thickness
    ## lets view the image
    plt_show(image)
    
#LOADING PRE-TRAINED CLASSIFIER
#Loading the pre-trained classifier from andrewssobral git repository
haarcascade_url = 'https://raw.githubusercontent.com/andrewssobral/vehicle_detection_haarcascades/master/cars.xml'
haar_name = "cars.xml"
urllib.request.urlretrieve(haarcascade_url, haar_name)
detector = cv2.CascadeClassifier(haar_name)

#Trying out on a single image
## we will read in a sample image
image_url = "https://s3.us.cloud-object-storage.appdomain.cloud/cf-courses-data/CognitiveClass/CV0101/Dataset/car-road-behind.jpg"
image_name = "car-road-behind.jpg"
urllib.request.urlretrieve(image_url, image_name)
image = cv2.imread(image_name)

#Lets have a look a this image
plt_show(image)

#Lets run the function on it
detect_obj(image)
