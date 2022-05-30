#Applying geometric operations (scaling, translation, rotation) to an image
#Applying mathematical operations (array and matrix operations) to an image

#Load necessary packages and functions
import matplotlib.pyplot as plt #Matplotlib for plotting images
from PIL import Image #Pillow Image functions
import numpy as np #Numpy for arrays

#Download necessary images
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/lenna.png -O lenna.png
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/baboon.png -O baboon.png
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/barbara.png -O barbara.png  
  
#Helper function to plot two images next to each other
#Helpful in visualizing changes we made to the images
def plot_image(image_1, image_2,title_1="Orignal",title_2="New Image"):
    plt.figure(figsize=(10,10))
    plt.subplot(1, 2, 1)
    plt.imshow(image_1,cmap="gray")
    plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(image_2,cmap="gray")
    plt.title(title_2)
    plt.show()
    
#GEOMETRIC TRANSFORMATIONS

#Have alook at the image we will be working with
image = Image.open("lenna.png")
plt.imshow(image)
plt.show()

#RESIZING
#Resizing an image (width)
width, height = image.size #find the current size
new_width = 2 * width #set new width to double original width
new_hight = height #set new heigh to original height
new_image = image.resize((new_width, new_hight)) #use PILs resize function on image
plt.imshow(new_image)
plt.show()

#Resizing an image (height)
new_width = width
new_hight = 2 * height
new_image = image.resize((new_width, new_hight))
plt.imshow(new_image)
plt.show()

#Resizing an image (width + height)
new_width = 2 * width
new_hight = 2 * height
new_image = image.resize((new_width, new_hight))
plt.imshow(new_image)
plt.show()

#Resizing an image (making smaller)
new_width = width // 2
new_hight = height // 2
new_image = image.resize((new_width, new_hight))
plt.imshow(new_image)
plt.show()

#ROTATING
theta = 45 #setting angle
new_image = image.rotate(theta) #use PIL rotate function on image
plt.imshow(new_image)
plt.show()

#ARRAY TRANSFORMATIONS
image = np.array(image) #converting the image into a numpy array

#making the image intensity values higher by adding a constant value to the array
new_image = image + 20
plt.imshow(new_image)
plt.show()

#or multiplying each pixel by a constant value
new_image = 10 * image
plt.imshow(new_image)
plt.show()

#generating some random noise to the image 
Noise = np.random.normal(0, 20, (height, width, 3)).astype(np.uint8) #random normal distribution between 0 and 20, of dimensions height x width x 3 colour channels
Noise.shape #check that the shape is correct
new_image = image + Noise #add the two arrays together
plt.imshow(new_image)
plt.show()

#adding noise multiplicatelvy
new_image = image*Noise
plt.imshow(new_image)
plt.show()

#MATRIX OPERATIONS
im_gray = Image.open("barbara.png") #even though this image is gray (and thus only requires 1 channel, it has three channels)
from PIL import ImageOps 
im_gray = ImageOps.grayscale(im_gray) #make it greyscale
im_gray = np.array(im_gray)
plt.imshow(im_gray,cmap='gray')
plt.show()

#we can decompose the image into its channels
U, s, V = np.linalg.svd(im_gray , full_matrices=True)
s.shape

#converting s into S
S = np.zeros((im_gray.shape[0], im_gray.shape[1]))
S[:image.shape[0], :image.shape[0]] = np.diag(s)

#plotting
plot_image(U, V, title_1="Matrix U", title_2="Matrix V")

#plotting S
plt.imshow(S, cmap='gray')
plt.show() #most elements in S are 0

#find the matrix product of all the matrices. 
#First, perform matrix multiplication on S and U and assign it to B and plot the results:
B = S.dot(V)
plt.imshow(B,cmap='gray')
plt.show()

#Then have A be the multiplication of B with U
A = U.dot(B)

#looks still good!
plt.imshow(A,cmap='gray')
plt.show()

#quality reduction works
for n_component in [1,10,100,200, 500]: #we will this list from 1 to 500 components
    S_new = S[:, :n_component] #everything, everything until n
    V_new = V[:n_component, :] #everything until n, everything
    A = U.dot(S_new.dot(V_new)) #dot products
    plt.imshow(A,cmap='gray')
    plt.title("Number of Components:"+str(n_component))
    plt.show()
    
#this is useful in order to  pre-process images. 
#In case you have HUGE image file sizes, we have to see whether we can reduce the file size by doing this quality reduction
