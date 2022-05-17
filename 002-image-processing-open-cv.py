import os #necessary for working directory function
import cv2 #library for computer vision. more functions than PIL, but more difficult to use
import matplotlib as plt #library for visualization

#Downloading necessary images (via console)
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/lenna.png -O lenna.png
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/baboon.png -O baboon.png
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/barbara.png -O barbara.png  
  
#helper function to put images side-by-side
def get_concat_h(im1, im2):
    #https://note.nkmk.me/en/python-pillow-concat-images/
    dst = Image.new('RGB', (im1.width + im2.width, im1.height)) #create a new image container as wide as both inputs, as high as input 2 (should actually be the max between img 1 and 2)
    dst.paste(im1, (0, 0)) #plug image 1 at the top left
    dst.paste(im2, (im1.width, 0)) #plug image 2 in the top middle
    return dst
  
#define variable as the image name I just downloaded into my working directory
my_image = "lenna.png"

#what's my current working directory?
cwd = os.getcwd()
cwd 

#define the image path of the image I downloaded
image_path = os.path.join(cwd, my_image)
image_path

#load the image into memory
image = cv2.imread(my_image)

#result is a numpy array with intensity values
type(image)

#shape is the same as when done by PIL. However, PIL returns three values for each pixel of R, G, B; while CV returns three values for each pixel of B, G, R
image.shape

#we can phyiscally see the image using opencv's imshow function (can break Jupyter Notebooks) or using matplotlib library
#cv2.imshow('image', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

plt.figure(figsize=(10,10))
plt.imshow(image)
plt.show()

#since matplotlib expects RGB and not BGR, we can convert the image
new_image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10,10))
plt.imshow(new_image)
plt.show()

#if the image is not in the working directory, access it using its path
image = cv2.imread(image_path)
image.shape

#save the image as jpeg
cv2.imwrite("lenna.jpg", image)

#we can convert images to grayscale using
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#this new image only has 1 value for each pixel, namely the intensity value of whiteness
image_gray.shape

#we can plot this new image, put have to specifiy that the input is gray
plt.figure(figsize=(10, 10))
plt.imshow(image_gray, cmap='gray')
plt.show()

#saving the new greyscale image:
cv2.imwrite('lena_gray_cv.jpg', image_gray)

#alternatively, while loading, we can also set as greyscale
im_gray = cv2.imread('barbara.png', cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(10,10))
plt.imshow(im_gray,cmap='gray')
plt.show()

#we can work with different color channels
baboon=cv2.imread('baboon.png')
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))
plt.show()

#let us assign the color channels
blue, green, red = baboon[:, :, 0], baboon[:, :, 1], baboon[:, :, 2]

#concatonating image channels on top of each other
im_bgr = cv2.vconcat([blue, green, red])

#this allows us to understand which image channels have the most intensity
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))
plt.title("RGB image")
plt.subplot(122)
plt.imshow(im_bgr,cmap='gray')
plt.title("Different color channels  blue (top), green (middle), red (bottom)  ")
plt.show()

#we can slice images, e.g., just looking at the top half
rows = 256
plt.figure(figsize=(10,10))
plt.imshow(new_image[0:rows,:,:])
plt.show()

#or the left half
columns = 256
plt.figure(figsize=(10,10))
plt.imshow(new_image[:,0:columns,:])
plt.show()

#we can reassign the array to a new variable
A = new_image.copy()
plt.imshow(A)
plt.show()

#do not just manipulate the values in the array of the original variable, or else you'll be basically deleting the original image (see below)
B = A
A[:,:,:] = 0
plt.imshow(B)
plt.show()

#we can copy an image and manipulate it to only show the red channel
baboon_red = baboon.copy()
baboon_red[:, :, 0] = 0
baboon_red[:, :, 1] = 0
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(baboon_red, cv2.COLOR_BGR2RGB))
plt.show()

#only the blue channel
baboon_blue = baboon.copy()
baboon_blue[:, :, 1] = 0
baboon_blue[:, :, 2] = 0
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(baboon_blue, cv2.COLOR_BGR2RGB))
plt.show()

#only the green channel
baboon_green = baboon.copy()
baboon_green[:, :, 0] = 0
baboon_green[:, :, 2] = 0
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(baboon_green, cv2.COLOR_BGR2RGB))
plt.show()

#everything apart from blue channel
baboon_without_blue = baboon.copy()
baboon_without_blue[:, :, 0] = 0
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(baboon_without_blue, cv2.COLOR_BGR2RGB))
plt.show()
