#As stated OpenCV is more powerful, but more difficult to use than PIL
#Copying, flipping, cropping, pixel manipulation

import matplotlib.pyplot as plt #for visualization
import cv2 #for manipulation
import numpy as np #for manipulation

#Downloading images in the console
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/cat.png -O cat.png
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/lenna.png -O lenna.png
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/baboon.png -O baboon.png
  
#Copying images to prevent aliasing 
baboon = cv2.imread("baboon.png")
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))
plt.show()

#because otherwise you'll have this problem
A = baboon
id(A)==id(baboon)
id(A)

B = baboon.copy()
id(B)==id(baboon)

baboon[:,:,] = 0
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))
plt.title("baboon")
plt.subplot(122)
plt.imshow(cv2.cvtColor(A, cv2.COLOR_BGR2RGB))
plt.title("array A")
plt.show()

#so, instead, we want to have a copy:
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))
plt.title("baboon")
plt.subplot(122)
plt.imshow(cv2.cvtColor(B, cv2.COLOR_BGR2RGB))
plt.title("array B")
plt.show()

#FLIPPING IMAGES#
#using indexing

image = cv2.imread("cat.png")
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

#create a new empty shape
width, height,C=image.shape
print('width, height,C',width, height,C)
array_flip = np.zeros((width, height,C),dtype=np.uint8)

#flipping
for i,row in enumerate(image):
        array_flip[width-1-i,:,:]=row
    
plt.figure(figsize=(5,5))
plt.imshow(cv2.cvtColor(array_flip, cv2.COLOR_BGR2RGB))
plt.show()

#using Open CV functions instead

#flipcode, three different applications namely flipcode 0, which is vertical, flipcode 1 which is horizontal and flipcode 2 which is both
for flipcode in [0,1,-1]:
    im_flip =  cv2.flip(image,flipcode )
    plt.imshow(cv2.cvtColor(im_flip,cv2.COLOR_BGR2RGB))
    plt.title("flipcode: "+str(flipcode))
    plt.show()
 

#we can rotate using open cv too
im_flip = cv2.rotate(image,0)
plt.imshow(cv2.cvtColor(im_flip,cv2.COLOR_BGR2RGB))
plt.show()

#these are some of the builtin attributes: Normally, they are just assigned a number
flip = {"ROTATE_90_CLOCKWISE":cv2.ROTATE_90_CLOCKWISE,
        "ROTATE_90_COUNTERCLOCKWISE":cv2.ROTATE_90_COUNTERCLOCKWISE,
        "ROTATE_180":cv2.ROTATE_180}

flip["ROTATE_90_CLOCKWISE"] #see what is the value of this key

#let us plot all three in our flip dictionary
for key, value in flip.items():
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("orignal")
    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(cv2.rotate(image,value), cv2.COLOR_BGR2RGB))
    plt.title(key)
    plt.show()
    
    
#CROPPING and PIXEL MANIPULATION#
#using index slicing

upper = 150
lower = 400
crop_top = image[upper: lower,:,:]
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(crop_top, cv2.COLOR_BGR2RGB))
plt.show()

left = 150
right = 400
crop_horizontal = crop_top[: ,left:right,:]
plt.figure(figsize=(5,5))
plt.imshow(cv2.cvtColor(crop_horizontal, cv2.COLOR_BGR2RGB))
plt.show()

array_sq = np.copy(image)
array_sq[upper:lower,left:right,:] = 0
plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.title("orignal")
plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(array_sq,cv2.COLOR_BGR2RGB))
plt.title("Altered Image")
plt.show()

#using OpenCV
start_point, end_point = (left, upper),(right, lower)
image_draw = np.copy(image)
cv2.rectangle(image_draw, pt1=start_point, pt2=end_point, color=(0, 255, 0), thickness=3) 
plt.figure(figsize=(5,5))
plt.imshow(cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB))
plt.show()

#adding text using OPenCV
#img: Image array
#text: Text string to be overlayed
#org: Bottom-left corner of the text string in the image
#fontFace: tye type of font
#fontScale: Font scale
#color: Text color
#: Thickness of the lines used to draw a text
#lineType: Line type
image_draw=cv2.putText(img=image,text='Cat Face Detected',org=(10,500),color=(255,255,255),fontFace=4,fontScale=5,thickness=2)
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(image_draw,cv2.COLOR_BGR2RGB))
plt.show()
