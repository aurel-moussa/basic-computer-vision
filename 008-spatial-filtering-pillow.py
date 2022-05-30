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

