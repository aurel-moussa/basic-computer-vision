#Pixel Transforms are operations you perform one pixel at a time. 
#Histograms display the intensity of the image and can be used to optimize image characteristics. 
#Intensity Transformations, making objects easier to see by improving image contrast and brightness.
#Thresholding to segment objects from images.

import matplotlib.pyplot as plt #for visualization
import cv2 #for manipulation
import numpy as np #for manipulation

#get the images via console
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/lenna.png -O lenna.png
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/baboon.png -O baboon.png
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/goldhill.bmp -O goldhill.bmp
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/cameraman.jpeg -O cameraman.jpeg
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/zelda.png -O zelda.png
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/mammogram.png -O mammogram.png
  
#helper function to put images side-by-side
def plot_image(image_1, image_2,title_1="Orignal", title_2="New Image"):
    plt.figure(figsize=(10,10))
    plt.subplot(1, 2, 1)
    plt.imshow(image_1,cmap="gray")
    plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(image_2,cmap="gray")
    plt.title(title_2)
    plt.show()
    
#helper function to plot histograms of image intensities side-by-side
def plot_hist(old_image, new_image,title_old="Orignal", title_new="New Image"):
    intensity_values=np.array([x for x in range(256)])
    plt.subplot(1, 2, 1)
    plt.bar(intensity_values, cv2.calcHist([old_image],[0],None,[256],[0,256])[:,0],width = 5)
    plt.title(title_old)
    plt.xlabel('intensity')
    plt.subplot(1, 2, 2)
    plt.bar(intensity_values, cv2.calcHist([new_image],[0],None,[256],[0,256])[:,0],width = 5)
    plt.title(title_new)
    plt.xlabel('intensity')
    plt.show()
    
#cv.calcHist will be our histogram tool for images
#cv2.calcHist(CV array:[image] 
#this is the image channel:[0],
#for this course it will always be [None],
#the number of bins:[L],
#the range of index of bins:[0,L-1])

#let's create a toy/model image to see histogram in action
toy_image = np.array([[0,2,2],[1,1,1],[1,1,2]],dtype=np.uint8)
plt.imshow(toy_image, cmap="gray")
plt.show()
print("toy_image:",toy_image)

#using normal histogram, it should look like htis
plt.bar([x for x in range(6)],[1,5,2,0,0,0]) 
plt.show()

#let's try using calcHist
goldhill = cv2.imread("goldhill.bmp",cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(10,10))
plt.imshow(goldhill,cmap="gray")
plt.show()

#calculate its histogram
hist = cv2.calcHist([goldhill],[0], None, [256], [0,256]) #we set 256 bins, from the range 0 to 256
intensity_values = np.array([x for x in range(hist.shape[0])]) #we take the intensity values that exist in this image
plt.bar(intensity_values, hist[:,0], width = 5) #and we plot the intensitvy values against the histogram
plt.title("Bar histogram")
plt.show()

#the probability mass function of this image
PMF = hist / (goldhill.shape[0] * goldhill.shape[1])

#continuous diagram
plt.plot(intensity_values,hist)
plt.title("histogram")
plt.show()

#histogram for each individual colour channel
baboon = cv2.imread("baboon.png")
plt.imshow(cv2.cvtColor(baboon,cv2.COLOR_BGR2RGB))
plt.show()

color = ('blue','green','red')
for i,col in enumerate(color):
    histr = cv2.calcHist([baboon],[i],None,[256],[0,256])
    plt.plot(intensity_values,histr,color = col,label=col+" channel")
    plt.xlim([0,256])
plt.legend()
plt.title("Histogram Channels")
plt.show()

#INTENSITY VALUES TRANSFORMATIONS

#using array transformations reverse colours
neg_toy_image = -1 * toy_image + 255
#for example, let us say we have an image which has three pixels, white, grey and black. 
#white has normally a value of 255. times negatives 1 plus 255 = 0
#black has normally a value of 0. times negative 1 plus 255 = 255
#grey has normally a value of 130. times negative 1 plus 255 = 116

print("toy image\n", neg_toy_image)
print("image negatives\n", neg_toy_image)

plt.figure(figsize=(10,10))
plt.subplot(1, 2, 1) 
plt.imshow(toy_image,cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(neg_toy_image,cmap="gray")
plt.show()
print("toy_image:",toy_image)

#this helps, for example, in mammograms
image = cv2.imread("mammogram.png", cv2.IMREAD_GRAYSCALE)
cv2.rectangle(image, pt1=(160, 212), pt2=(250, 289), color = (255), thickness=2) 
plt.figure(figsize = (10,10))
plt.imshow(image, cmap="gray")
plt.show()

img_neg = -1 * image + 255
plt.figure(figsize=(10,10))
plt.imshow(img_neg, cmap = "gray")
plt.show()

#thinking of brighteness (beta) and contrast (alpha)
#if our original image is f(x,y), a transformed image g(x,y) would be done via alpha*f(x,y)+beta

#we could do this via array operations, but easier to use CV function
alpha = 1 # Simple contrast control
beta = 100   # Simple brightness control   
new_image = cv2.convertScaleAbs(goldhill, alpha=alpha, beta=beta)
plot_image(goldhill, new_image, title_1 = "Orignal", title_2 = "brightness control") #our helper function from before

#see the shift in the histogram
plt.figure(figsize=(10,5))
plot_hist(goldhill, new_image, "Orignal", "brightness control") #our helper function fom before

#we can also increase contrast by affecting the alpha
plt.figure(figsize=(10,5))
alpha = 2# Simple contrast control
beta = 0 # Simple brightness control   # Simple brightness control
new_image = cv2.convertScaleAbs(goldhill, alpha=alpha, beta=beta)
plot_image(goldhill,new_image,"Orignal","contrast control")

plt.figure(figsize=(10,5))
plot_hist(goldhill, new_image,"Orignal","contrast control")

#and we can also affect contrast and decrease brightness at the same time
plt.figure(figsize=(10,5))
alpha = 3 # Simple contrast control
beta = -200  # Simple brightness control   
new_image = cv2.convertScaleAbs(goldhill, alpha=alpha, beta=beta)
plot_image(goldhill, new_image, "Orignal", "brightness & contrast control")
plt.figure(figsize=(10,5))
plot_hist(goldhill, new_image, "Orignal", "brightness & contrast control")

#There are also non-linear transformations of these, as well as self-adjusting ones

#Histogram Equalization being one of these: #welcome to Instagram Filters
zelda = cv2.imread("zelda.png",cv2.IMREAD_GRAYSCALE)
new_image = cv2.equalizeHist(zelda)
plot_image(zelda,new_image,"Orignal","Histogram Equalization")

#we can also use thresholding, for example, to manipulate an image which has text on it to get rid of smudges etc, and only focus on the text
#basically, we will assign each pixel to either show as white or black (usually), depending on whether they are lower or higher than some threshold

def thresholding(input_img,threshold,max_value=255, min_value=0):
    N,M=input_img.shape
    image_out=np.zeros((N,M),dtype=np.uint8)
        
    for i  in range(N):
        for j in range(M):
            if input_img[i,j]> threshold:
                image_out[i,j]=max_value
            else:
                image_out[i,j]=min_value
                
    return image_out   
  
#let us apply this function to our toy/model image
threshold = 1
max_value = 2
min_value = 0
thresholding_toy = thresholding(toy_image, threshold=threshold, max_value=max_value, min_value=min_value)
thresholding_toy

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(toy_image, cmap="gray")
plt.title("Original Image")
plt.subplot(1, 2, 2)
plt.imshow(thresholding_toy, cmap="gray")
plt.title("Image After Thresholding")
plt.show()

#let us have a look at a photo
image = cv2.imread("cameraman.jpeg", cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(10, 10))
plt.imshow(image, cmap="gray")
plt.show()

hist = cv2.calcHist([image], [0], None, [256], [0, 256])
plt.bar(intensity_values, hist[:, 0], width=5)
plt.title("Bar histogram")
plt.show()

#cameraman corresponds to the intensity level at and below 90
threshold = 87
max_value = 255
min_value = 0
new_image = thresholding(image, threshold=threshold, max_value=max_value, min_value=min_value)

plot_image(image, new_image, "Orignal", "Image After Thresholding")

plt.figure(figsize=(10,5))
plot_hist(image, new_image, "Orignal", "Image After Thresholding")

#instead of using the thresholding function we just created above, we can also use CV function for this
#cv.threshold(grayscale image, threshold value, maximum value to use, thresholding type )
#thresholding types are stuff like cv2.THRESH_BINARY for a binary endproduct
#cv2.THRESH_TRUNC will not change the values if the pixels are less than the threshold value


ret, new_image = cv2.threshold(image,threshold,max_value,cv2.THRESH_BINARY) 
#ret is the threshold value and new_image is the image after thresholding has been applied
plot_image(image,new_image,"Orignal","Image After Thresholding")
plot_hist(image, new_image,"Orignal","Image After Thresholding")

#we can also automize the threshold and maxvalue
ret, otsu = cv2.threshold(image,0,255,cv2.THRESH_OTSU)
plot_image(image,otsu,"Orignal","Otsu")
plot_hist(image, otsu,"Orignal"," Otsu's method")

