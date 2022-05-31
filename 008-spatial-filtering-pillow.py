#Spatial filtering use pixels in a neighborhood to determine the present pixel value
#Some applications include filtering and sharpening
#Used in many steps in computer vision, such as segmentation, and are a key building block in Artificial Intelligence algorithms

#Looking at linear filtering (filtering noise, Gaussian blur, image sharpening), edge-finding, and using median to set pixel value

#importing necessary libraries
import matplotlib.pyplot as plt #plotting and viewing images
from PIL import Image #loading images
import numpy as np #create kernels for filtering

#Downloading test images (use the console)
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/cameraman.jpeg
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/lenna.png
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/barbara.png   
  
#Helper function to see original image and new image side-by-side
def plot_image(image_1, image_2,title_1="Orignal",title_2="New Image"):
    plt.figure(figsize=(10,10))
    plt.subplot(1, 2, 1)
    plt.imshow(image_1)
    plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(image_2)
    plt.title(title_2)
    plt.show()
    
#LINEAR FILTERING
#Filtering involves enhancing an image by, e.g., removing the noise from an image. Noise can be caused by a bad camera or bad image compression. 
#The same factors that cause noise may lead to blurry images. We can apply filters to sharpen these images. 
#Convolution is a standard way to filter an image. The filter is called the kernel and different kernels perform different tasks. 
#In addition, Convolution is used for many of the most advanced artificial intelligence algorithms. 

#We simply take the dot product of the kernel and an equally-sized portion of the image. 
#We then shift the kernel (i.e. which area of the image it should work on) and repeat.

#LOADING AND CREATING NOISE
#Let us load a nice image
# Loads the image from the specified file
image = Image.open("lenna.png")
# Renders the image
plt.figure(figsize=(5,5))
plt.imshow(image)
plt.show()

#Let us make the image look grainy by adding white noise
# Get the number of rows and columns in the image
rows, cols = image.size
# Creates values using a normal distribution with a mean of 0 and standard deviation of 15, the values are converted to unit8 which means the values are between 0 and 255
noise = np.random.normal(0,15,(rows,cols,3)).astype(np.uint8)
# Add the noise to the image
noisy_image = image + noise
# Creates a PIL Image from an array
noisy_image = Image.fromarray(noisy_image)
# Plots the original image and the image with noise using the function defined at the top
plot_image(image, noisy_image, title_1="Orignal", title_2="Image Plus Noise")

#When adding noise to an image sometimes the value might be greater than 255, 
#in this case 256, is subtracted from the value to wrap the number around keeping it between 0 and 255. 
#For example, consider an image with an RGB value of 137 and we add noise with an RGB value of 215 
#to get an RGB value of 352. We then subtract 256, the total number of possible values between 0 and 255, to get a number between 0 and 255.

#FILTERING NOISE
from PIL import ImageFilter

#let us use smoothing filters. They average out pixels in a neighbourhood. Sometimes called low-pass filters. 
#mean filtering kernel simply averages ot the kernels in a neighbourhood

# Create a kernel which is a 5 by 5 array where each value is 1/36
#Note: The maximum kernel size with which PIL can work is 5x5... use Open CV instead if you require BIG BOY kernels
kernel = np.ones((5,5))/36
# Create a ImageFilter Kernel by providing the kernel size and a flattened kernel
kernel_filter = ImageFilter.Kernel((5,5), kernel.flatten())

# Filters the images using the kernel. The .filter function does a convolution between image and kernel on each colour channel independently
image_filtered = noisy_image.filter(kernel_filter)

#Let's have a looksey
# Plots the Filtered and Image with Noise using our trusty helper function
plot_image(image_filtered, noisy_image,title_1="Filtered image",title_2="Image Plus Noise")

#Decreasing the kernel size keeps the image sharp, but filters less noise
#Let us see this with a 3x3 kernel, instead of the 5x5 we did above

# Create a kernel which is a 3 by 3 array where each value is 1/36
kernel = np.ones((3,3))/36
# Create a ImageFilter Kernel by providing the kernel size and a flattened kernel
kernel_filter = ImageFilter.Kernel((3,3), kernel.flatten())
# Filters the images using the kernel
image_filtered = noisy_image.filter(kernel_filter)
# Plots the Filtered and Image with Noise using the function defined at the top
plot_image(image_filtered, noisy_image,title_1="Filtered image",title_2="Image Plus Noise")

#GAUSSIAN BLUR
# Filters the images using GaussianBlur pre-defined filter function
image_filtered = noisy_image.filter(ImageFilter.GaussianBlur) #radius parameter for Gaussian, default is 2
# Plots the Filtered Image then the Unfiltered Image with Noise
plot_image(image_filtered , noisy_image,title_1="Filtered image",title_2="Image Plus Noise")

# Filters the images using GaussianBlur on the image with noise using a 4 by 4 kernel 
image_filtered = noisy_image.filter(ImageFilter.GaussianBlur(4)) #parameter change
# Plots the Filtered Image then the Unfiltered Image with Noise
plot_image(image_filtered , noisy_image,title_1="Filtered image",title_2="Image Plus Noise")

#IMAGE SHARPENING
#Image sharpening is all about smoothing the image and calculating the derivatives
#Let us do this by the following kernel

# Common Kernel for image sharpening
kernel = np.array([[-1,-1,-1], 
                   [-1, 9,-1],
                   [-1,-1,-1]])
kernel = ImageFilter.Kernel((3,3), kernel.flatten())
# Applys the sharpening filter using kernel on the original image without noise
sharpened = image.filter(kernel)
# Plots the sharpened image and the original image without noise
plot_image(sharpened , image, title_1="Sharpened image",title_2="Image")

#We can also use pre-defined filters for sharpening
# Sharpends image using predefined image filter from PIL
sharpened = image.filter(ImageFilter.SHARPEN)
# Plots the sharpened image and the original image without noise
plot_image(sharpened , image, title_1="Sharpened image",title_2="Image")

#EDGE DETECTION
#Edges are where pixel intensities change. 
#The Gradient of a function outputs the rate of change; we can approximate the gradient of a grayscale image with convolution.
#Edge detection is super-duper important for object recognition

#Loading the image
# Loads the image from the specified file
img_gray = Image.open('barbara.png')
# Renders the image from the array of data, notice how it is 2 diemensional instead of 3 diemensional because it has no colour
plt.imshow(img_gray ,cmap='gray')

#Let us first enhance the edges, so they are more easily picked up later
# Filters the images using EDGE_ENHANCE filter
img_gray = img_gray.filter(ImageFilter.EDGE_ENHANCE)
# Renders the enhanced image
plt.imshow(img_gray ,cmap='gray')

# Filters the images using FIND_EDGES filter
img_gray = img_gray.filter(ImageFilter.FIND_EDGES)
# Renders the filtered image
plt.figure(figsize=(10,10))
plt.imshow(img_gray ,cmap='gray')

#It makes a lot of sense to run an image through quality compression/reduction first (because of computational resources), 
#then edge enhancing, the edge finding, before inputting it as the arrays to use for object recognition

#MEDIAN FILTERING
#Median filters find the median of all the pixels underneath the kernel area. The central element of the image is then repalced with this median value
#We can use median filters to improve segmentation

# Load the camera man image
image = Image.open("cameraman.jpeg")
# Make the image larger when it renders
plt.figure(figsize=(10,10))
# Renders the image
plt.imshow(image,cmap="gray")

#Median filters blurs the background, increasing the segmentation between the image in the foreground (cameraboy) and the background
image = image.filter(ImageFilter.MedianFilter)
plt.figure(figsize=(10,10))
# Renders the image
plt.imshow(image,cmap="gray")

#This segmentation can be very useful for object deteection, to make certain objects stand out from other parts of the picture, such as the background
