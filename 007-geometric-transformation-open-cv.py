#Applying geometric operations (scaling, translation, rotation) to an image
#Applying mathematical operations (array and matrix operations) to an image

#Load necessary packages
import matplotlib.pyplot as plt #for image plotting
import cv2 #for image manipulation
import numpy as np #for array manipulation

#Download neceesary images (use console for this)
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/lenna.png -O lenna.png
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/baboon.png -O baboon.png
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/barbara.png -O barbara.png  
 
#Helper function for putting images side-by-side
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
#SCALING

#Let us create a toy/model image
toy_image = np.zeros((6,6)) #creating a 6x6 array with zeros
toy_image[1:5,1:5]=255 #make columns and rows 1-5 to 255 (white)
toy_image[2:4,2:4]=0 #make columns and rows 2-4 to 0 (black)
plt.imshow(toy_image,cmap='gray') #show this array as an image
plt.show()
toy_image

#Rescaling via fx and fy dimensions
new_toy = cv2.resize(toy_image,None,fx=2, fy=1, interpolation = cv2.INTER_NEAREST ) #interpolation sets the intensity values of the "newly" generated spaces
#INTER_NEAREST uses the nearest pixel and INTER_CUBIC uses several pixels near the pixel value we would like to estimate
plt.imshow(new_toy,cmap='gray')
plt.show()

#Let us try with a non-dummy image
#scale just x-axis (width)
image = cv2.imread("lenna.png")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) #ensuring that the BGR is converted into RGB first
plt.show()
new_image = cv2.resize(image, None, fx=2, fy=1, interpolation=cv2.INTER_CUBIC)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()
print("old image shape:", image.shape, "new image shape:", new_image.shape)

#scale just y-axis (height)
new_image = cv2.resize(image, None, fx=1, fy=2, interpolation=cv2.INTER_CUBIC)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()
print("old image shape:", image.shape, "new image shape:", new_image.shape)

#scale x and y-axes
new_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()
print("old image shape:", image.shape, "new image shape:", new_image.shape)

#shrink y-axis
new_image = cv2.resize(image, None, fx=1, fy=0.5, interpolation=cv2.INTER_CUBIC)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()
print("old image shape:", image.shape, "new image shape:", new_image.shape)

#specifc the number of rows and columns, instead
rows = 100
cols = 200
new_image = cv2.resize(image, (100, 200), interpolation=cv2.INTER_CUBIC)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()
print("old image shape:", image.shape, "new image shape:", new_image.shape)

#TRANSLATING
#fancy way of saying: moving an image within its canvas
tx = 100 #number of pixels to shift in x-direction
ty = 0 #number of pixels to shift in y-direction
M = np.float32([[1, 0, tx], [0, 1, ty]]) #creates a transformation matrix (needs to be converted into float values)
M

#shape of our image
rows, cols, _ = image.shape

#We use the function warpAffine from the cv2 module. 
#The first input parater is an image array, 
#the second input parameter is the transformation matrix M, 
#and the final input paramter is the length and width of the output image (cols, rows)

new_image = cv2.warpAffine(image, M, (cols, rows))
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()

#the above cuts the image (i.e. the canvas stays the same)
#if we also want to increase the canvas, we can add tx and ty to the cols and rows

new_image = cv2.warpAffine(image, M, (cols + tx, rows + ty))
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()

#let us shift vertically
tx = 0
ty = 50
M = np.float32([[1, 0, tx], [0, 1, ty]])
new_iamge = cv2.warpAffine(image, M, (cols + tx, rows + ty))
plt.imshow(cv2.cvtColor(new_iamge, cv2.COLOR_BGR2RGB))
plt.show()

#ROTATION
#We can use rotation function getRotationMatrix2D from Open CV
#needs 3 parameters:
#center: Center of the rotation in the source image. We will only use the center of the image
#angle: Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner)
#scale: Isotropic scale factor, in our example it will be one

theta = 45.0
M = cv2.getRotationMatrix2D(center=(3, 3), angle=theta, scale=1) #create the transformation matrix
new_toy_image = cv2.warpAffine(toy_image, M, (6, 6))

plot_image(toy_image, new_toy_image, title_1="Orignal", title_2="rotated image")
new_toy_image  #image in array shows that many of the values have been interpolated

#Let us do the same rotation transformation on a real picture
cols, rows, _ = image.shape
M = cv2.getRotationMatrix2D(center=(cols // 2 - 1, rows // 2 - 1), angle=theta, scale=1)
new_image = cv2.warpAffine(image, M, (cols, rows))
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()

#ARRAY OPERATIONS
#Array Manipulation is the same as with Pillow, as this mostly uses inbuilt Python and Numpy functions
new_image = image + 20
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()

new_image = 10 * image
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()

#Adding noise
Noise = np.random.normal(0, 20, (rows, cols, 3)).astype(np.uint8)
Noise.shape

new_image = image + Noise
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()

#Mulitplying image by noise
new_image = image*Noise
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()

#MATRIX OPERATIONS
#Loading a grayscale image
im_gray = cv2.imread('barbara.png', cv2.IMREAD_GRAYSCALE)
im_gray.shape
plt.imshow(im_gray,cmap='gray')
plt.show()

#We can apply algorithms designed for matrices such as Singular Value Decomposition
#decomposing our image matrix into a product of three matrices.
U, s, V = np.linalg.svd(im_gray , full_matrices=True)

#s is not rectangular
s.shape

#convert s into diagonal matrix S
S = np.zeros((im_gray.shape[0], im_gray.shape[1])) #create a matrix of zeroes, with width of the im_gray and height of the im_gray
S[:image.shape[0], :image.shape[0]] = np.diag(s) #set S from 0 to width and from 0 to height, to the diagonal matrix of s
S.shape #it has a good shape

#plot the matrices of U and V
plot_image(U,V,title_1="Matrix U ",title_2="matrix  V")

#plot matrix S, seing it is mostly black
plt.imshow(S,cmap='gray')
plt.show()

#create a matrix product
B = S.dot(V)
plt.imshow(B,cmap='gray')
plt.show()

A = U.dot(B)

#and have a look at the image
plt.imshow(A,cmap='gray')
plt.show()

#we can also further reduce the image size by reducing the components
for n_component in [1,10,100,200, 500]:
    S_new = S[:, :n_component]
    V_new = V[:n_component, :]
    A = U.dot(S_new.dot(V_new))
    plt.imshow(A,cmap='gray')
    plt.title("Number of Components:"+str(n_component))
    plt.show()

