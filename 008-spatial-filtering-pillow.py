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
