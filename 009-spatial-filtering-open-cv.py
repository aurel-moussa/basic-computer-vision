#Again, spatial operations are operations that use the values of pixels in a neighbourhood to determine a specific pixel value
#One can use these operations for filtering, sharpening, edge detection, segmentation; all important in AI vision

#We'll look at linear filtering (filtering noise, gaussian blur, image sharpening),
#edge detection
#median filtering
#and threshold filtering (new with Open CV!)

#Get packages
# Used to view the images
import matplotlib.pyplot as plt
# Used to perform filtering on an image
import cv2
# Used to create kernels for filtering
import numpy as np

#Helper function to plot images side-by-side
def plot_image(image_1, image_2,title_1="Orignal",title_2="New Image"):
    plt.figure(figsize=(10,10))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB))
    plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB))
    plt.title(title_2)
    plt.show()

#Let's get our images (in the console)
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/cameraman.jpeg
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/lenna.png
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/barbara.png   
  
#LINEAR FILTERING
#Filtering is the manipulation of an image, e.g., removing image noise
#Image noise ihappens with a bad camera or bad image compression
#These factors may also lead to blurry images
#We can apply filters to remove noise and sharpen images
#Convolution is a standard way to filter an image
#Filter is called the kernel; different kernels perform different tasks
#Convolution is also used in many advanced AI algorithms
#We take the dot product of the kernel and an equally sized portion of the image
#We shift the area in which the kernel operates and repeat

#Let's load an image
# Loads the image from the specified file
image = cv2.imread("lenna.png")
print(image)
# Converts the order of the color from BGR (Blue Green Red) to RGB (Red Green Blue) then renders the image from the array of data
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#Let's apply some grainyness
# Get the number of rows and columns in the image
rows, cols,_= image.shape
# Creates values using a normal distribution with a mean of 0 and standard deviation of 15, the values are converted to unit8 which means the values are between 0 and 255
noise = np.random.normal(0,15,(rows,cols,3)).astype(np.uint8) #rows, columns and 3 because we have three colour channels
# Add the noise to the image
noisy_image = image + noise
# Plots the original image and the image with noise using the function defined at the top
plot_image(image, noisy_image, title_1="Orignal",title_2="Image Plus Noise")
