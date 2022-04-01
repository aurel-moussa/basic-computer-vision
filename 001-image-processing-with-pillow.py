##INTRODUCTION##

#What is meant by image processing and computer vision?

#Image processing and computer vision tasks include displaying, 
#cropping, flipping, rotating, image segmentation, classification, 
#image restoration, image recognition, image generation. 
#Also, working with images via the cloud requires storing, transmitting, and gathering images through the internet.

#Why image processing and computer vision with Python?

#Python has many image processing tools, computer vision and artificial intelligence libraries. 
#Finally, it has many libraries for working with files in the cloud and on the internet.

#What this code does

#Working with image files using Pillow library (PIL) 
#Python Image Libraries
#Image Files and Paths
#Load in Image in Python
#Plotting an Image
#Gray Scale Images, Quantization and Color Channels
#PIL Images into NumPy Arrays

##CODE##

#01. Image Loading and File Handling

#Let us start by downloading some images (Note: the following is not Python code, but a UNIX command to get some images)
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/lenna.png -O lenna.png
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/baboon.png -O baboon.png
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/barbara.png -O barbara.png  

#Helper function to put two images next to each other. This will help with visualization later on.
def get_concat_h(image1, image2):
    #https://note.nkmk.me/en/python-pillow-concat-images/
    dst = Image.new('RGB', (image1.width + image2.width, image1.height)) 
    #Create a new RGB image which is as wide as image 1 and 2 together, and as heigh as image 1
    dst.paste(image1, (0, 0)) #Paste image 1 at the top left corner
    dst.paste(image2, (image1.width, 0)) #Paste image 2 at the top middle
    return dst

#Define my_image as the filename of the image
my_image = "lenna.png"

#Define the path of where your file is located
import os
cwd = os.getcwd() #current working directory
cwd 

#Get the full image path, with path and filename
image_path = os.path.join(cwd, my_image)
image_path

#Pillow (PIL) library is a popular library for loading images in Python. 
#Other libraries such as Keras and PyTorch use this library to work with images. 
#The Image module provides functions to load images from and saving images to the file system.

from PIL import Image
image = Image.open(my_image)
type(image)

#If working from an IDE such as Jupyter Notebooks, we can view the image directly by calling it
image #either directly
image.show() #or using the PIL show method

#Alternatively, we can use imshow method from matplotlib to show images
plt.figure(figsize=(10,10))
plt.imshow(image)
plt.show()

#If the image is not in the current working directory:
image = Image.open(image_path)

#Next up in this fabulous adventure: Some interesting image attributes
print(image.size) #pixels wide and pixels heigh
print(image.mode) #colour format (e.g., RGB)

#The `Image.open` method does not load image data into the computer memory. 
#The `load` method of `PIL` object reads the file content, decodes it, and expands the image into memory.
#I believe it should be closed immediately after using, or otherwise it takes up too much memory?

#Let us check the intensity of the (in this case RGB values) of the xth column and yth row
im = image.load()
x = 0
y = 1
im[y,x]

#Saving... (and converting!)
image.save("lenna.jpg")

#02. Image Processing

#ImageOps contains a lot of easy-to-use image processing operations. 
#Module may sometimes break, because it is not 100% stable (works best for grayscale or RGB)

from PIL import ImageOps 

#Convert our image into grayscale
image_gray = ImageOps.grayscale(image) 
image_gray 
image_gray.mode #L for Greyscale

#The quantization of an image means how many different (unique) intensity levels it has.
#For grayscale, this means how many different types of grey. Most images have 256 different levels.
#Levels can be decreased by using quantize method

image_gray.quantize(256 // 2)
#image_gray.show()
image_gray #For this example, the image quality still seems good

#Let us successively divide the amount of intensity levels, (and use our helpful helper function) to 
#see how the image quality is reduced
for n in range(3,8):
    plt.figure(figsize=(10,10))
    plt.imshow(get_concat_h(image_gray,  image_gray.quantize(256//2**n))) 
    plt.title("256 Quantization Levels  left vs {}  Quantization Levels right".format(256//2**n))
    plt.show()

#Amazing. Let us continue our magical journey with a different pic.

baboon = Image.open('baboon.png')
baboon

#Let us split up the colour channels RGB 
red, green, blue = baboon.split()

#Put the original image, next to the intensity levels of red, blue, green
get_concat_h(baboon, red)
get_concat_h(baboon, blue)
get_concat_h(baboon, green)

#03. Converting into arrays
#In order to later use machine learning functions, we will have to convert the image
