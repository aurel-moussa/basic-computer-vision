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

#FILTERING NOISE
#Let's use a smoothing filter or low pass filter to create the average of pixels in a neighbourhood

# Create a kernel which is a 6 by 6 array where each value is 1/36
kernel = np.ones((6,6))/36 #thank god, with OpenCV we can have kernels of size bigger than 5x5

#The function filter2D performs 2D convolution between the image src and the kernel on each colour channel independently
#The parameter "ddepth" has to do with the size of the output image, we will set it to -1 so the input and output are the same size

# Filters the images using the kernel
image_filtered = cv2.filter2D(src=noisy_image, ddepth=-1, kernel=kernel)

# Plots the Filtered and Image with Noise using the function defined at the top
plot_image(image_filtered, noisy_image,title_1="Filtered image",title_2="Image Plus Noise")

#A smaller kernel keeps image sharper, but filters less noise
# Creates a kernel which is a 4 by 4 array where each value is 1/16
kernel = np.ones((4,4))/16
# Filters the images using the kernel
image_filtered=cv2.filter2D(src=noisy_image,ddepth=-1,kernel=kernel)
# Plots the Filtered and Image with Noise using the function defined at the top
plot_image(image_filtered , noisy_image,title_1="filtered image",title_2="Image Plus Noise")

#GAUSSIAN BLUR
#Gaussian blur also filters noise, but generally is better at edge preservation
#Open CV parameters for the GaussianBlur function are:
#src = input image. The image can have many numbers of channels. These are processed indepdently
#ksize = Gaussian kernel size
#sigmaX = Gausian kernel standard deviation in the x direction
#sigmaY = Gaussian kernel standard deviation in the y direction. If sigmaY is 0, it will be equal to sigmaX

# Filters the images using GaussianBlur on the image with noise using a 4 by 4 kernel 
image_filtered = cv2.GaussianBlur(noisy_image,(5,5),sigmaX=4,sigmaY=4)
# Plots the Filtered Image then the Unfiltered Image with Noise
plot_image(image_filtered , noisy_image,title_1="Filtered image",title_2="Image Plus Noise")

#We can increase the GaussianBlur
#Sigma behaves like the size of the mean filter, 
#a larger value of sigma will make the image blurry, but you are still constrained by the size of the filter, there we set sigma to 10

# Filters the images using GaussianBlur on the image with noise using a 11 by 11 kernel 
image_filtered = cv2.GaussianBlur(noisy_image,(11,11),sigmaX=10,sigmaY=10)
# Plots the Filtered Image then the Unfiltered Image with Noise
plot_image(image_filtered , noisy_image,title_1="filtered image",title_2="Image Plus Noise")

#IMAGE SHARPENING
