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

#01. Image Loading
#Let us start by downloading some images (Note: the following is not Python code, but a UNIX command to get some images)
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/lenna.png -O lenna.png
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/baboon.png -O baboon.png
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/barbara.png -O barbara.png  

