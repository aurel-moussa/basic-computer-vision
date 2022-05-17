#Copying, flipping, cropping images, and changing specific pxiels

import matplotlib.pyplot as plt #for visualization
from PIL import Image #for manipulation
import numpy as np #for manipulation

#download image in console
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/cat.png -O cat.png
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/lenna.png -O lenna.png
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/baboon.png -O baboon.png
  
#copy image, as otherwise any manipulation you do is done on the original
baboon = np.array(Image.open('baboon.png'))
plt.figure(figsize=(5,5))
plt.imshow(baboon )
plt.show()

A = baboon
id(A) == id(baboon) #same memory location -> bad

B = baboon.copy()
id(B)==id(baboon) #different memory location ->good

#same memory location for A and baboon means this
baboon[:,:,] = 0
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(baboon)
plt.title("baboon")
plt.subplot(122)
plt.imshow(A)
plt.title("array A")
plt.show()

#different memory location for B and baboon means this
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(baboon)
plt.title("baboon")
plt.subplot(122)
plt.imshow(B)
plt.title("array B")
plt.show()

#flipping orientation fo the pixels
image = Image.open("cat.png")
plt.figure(figsize=(10,10))
plt.imshow(image)
plt.show()

#cast it into an array to see its shape
array = np.array(image)
width, height, C = array.shape
print('width, height, C', width, height, C)

#flipping an image without helper functions, ie just using array manipulation would be like this

array_flip = np.zeros((width, height, C), dtype=np.uint8) #same size, empty array

#assign the first row of pixels of the original array to the new array’s last row
#repeat the process for every row, incrementing the row number from the original array 
#decreasing the new array’s row index to assign the pixels accordingly.
#after excecuting the for loop, array_flip will become the flipped image.

for i,row in enumerate(array):
    array_flip[width - 1 - i, :, :] = row
    
#however, let us helper functions instead
from PIL import ImageOps

#flip function from PIL
im_flip = ImageOps.flip(image)
plt.figure(figsize=(5,5))
plt.imshow(im_flip)
plt.show()

#mirror function from PIL
im_mirror = ImageOps.mirror(image)
plt.figure(figsize=(5,5))
plt.imshow(im_mirror)
plt.show()

#transpose function from PIL
im_flip = image.transpose(1)
plt.imshow(im_flip)
plt.show()

#the Image module has built-in attributes for flips
flip = {"FLIP_LEFT_RIGHT": Image.FLIP_LEFT_RIGHT,
        "FLIP_TOP_BOTTOM": Image.FLIP_TOP_BOTTOM,
        "ROTATE_90": Image.ROTATE_90,
        "ROTATE_180": Image.ROTATE_180,
        "ROTATE_270": Image.ROTATE_270,
        "TRANSPOSE": Image.TRANSPOSE, 
        "TRANSVERSE": Image.TRANSVERSE}

#so, the flip left right is assigned to number 0
flip["FLIP_LEFT_RIGHT"]

#see each of the ouputs like this
for key, values in flip.items():
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title("orignal")
    plt.subplot(1,2,2)
    plt.imshow(image.transpose(values))
    plt.title(key)
    plt.show()

#let us continue image manipulation with cropping
#cropping using array slicing
upper = 150
lower = 400
crop_top = array[upper: lower,:,:]
plt.figure(figsize=(5,5))
plt.imshow(crop_top)
plt.show()

#continue cropping
left = 150
right = 400
crop_horizontal = crop_top[: ,left:right,:]
plt.figure(figsize=(5,5))
plt.imshow(crop_horizontal)
plt.show()

#we can crop using matplotlibs crop() method instead
image = Image.open("cat.png")
crop_image = image.crop((left, upper, right, lower))
plt.figure(figsize=(5,5))
plt.imshow(crop_image)
plt.show()

#let's flip the cropped image
crop_image = crop_image.transpose(Image.FLIP_LEFT_RIGHT)
crop_image

#we can change individual pixels using array indexing, again
array_sq = np.copy(array)
array_sq[upper:lower, left:right, 1:2] = 0 #set all Green and Blue things to 0 in this area
plt.figure(figsize=(5,5))
plt.subplot(1,2,1)
plt.imshow(array)
plt.title("orignal")
plt.subplot(1,2,2)
plt.imshow(array_sq)
plt.title("Altered Image")
plt.show()

#we can also use PILs ImageDraw module
from PIL import ImageDraw 
image_draw = image.copy()

#The draw constructor creates an object that can be used to draw in the given image. The input im is the image we would like to draw in.
image_fn = ImageDraw.Draw(im=image_draw)

#Drawing a rectangle
shape = [left, upper, right, lower] 
image_fn.rectangle(xy=shape,fill="red")

plt.figure(figsize=(10,10))
plt.imshow(image_draw)
plt.show()

#Apart from rectangles, we can also add text
from PIL import ImageFont
image_fn.text(xy=(0,0),text="box",fill=(0,0,0))

plt.figure(figsize=(10,10))
plt.imshow(image_draw)
plt.show()

#overlaying one image over another
image_lenna = Image.open("lenna.png")
array_lenna = np.array(image_lenna)
array_lenna[upper:lower,left:right,:]=array[upper:lower,left:right,:] #array is the cat picture data from above
plt.imshow(array_lenna)
plt.show()

#instead of doing it with indexing, we can also use the paste method instead
image_lenna.paste(crop_image, box=(left,upper))
plt.imshow(image_lenna)
plt.show()

