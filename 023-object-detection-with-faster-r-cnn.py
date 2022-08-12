#Faster R-CNN is a method for object detection that uses region proposal.
#We will use Faster R-CNN pre-trained on the coco dataset
#learn how to detect several objects by name and to use the likelihood of the object prediction being correct

#Downloading images (in console)
! wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-Coursera/images%20/images_part_5/DLguys.jpeg
! wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-Coursera/images%20/images_part_5/watts_photos2758112663727581126637_b5d4d192d4_b.jpeg
! wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-Coursera/images%20/images_part_5/istockphoto-187786732-612x612.jpeg
! wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-Coursera/images%20/images_part_5/jeff_hinton.png
  
#Installing/up-dating deep-learning libraries
! conda install pytorch=1.1.0 torchvision -c pytorch -y #old version of Pytorch
import torchvision
from torchvision import  transforms 
import torch
from torch import no_grad

import requests
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#Helper function
def get_predictions(pred,threshold=0.8,objects=None ):
    """
    This function will assign a string name to a predicted class and eliminate predictions whose likelihood  is under a threshold 
    
    pred: a list where each element contains a tuple that corresponds to information about  the different objects; Each element includes a tuple with the class yhat, probability of belonging to that class and the coordinates of the bounding box corresponding to the object 
    image : frozen surface
    predicted_classes: a list where each element contains a tuple that corresponds to information about  the different objects; Each element includes a tuple with the class name, probability of belonging to that class and the coordinates of the bounding box corresponding to the object 
    thre
    """


    predicted_classes= [(COCO_INSTANCE_CATEGORY_NAMES[i],p,[(box[0], box[1]), (box[2], box[3])]) for i,p,box in zip(list(pred[0]['labels'].numpy()),pred[0]['scores'].detach().numpy(),list(pred[0]['boxes'].detach().numpy()))]
    predicted_classes=[  stuff  for stuff in predicted_classes  if stuff[1]>threshold ]
    
    if objects  and predicted_classes :
        predicted_classes=[ (name, p, box) for name, p, box in predicted_classes if name in  objects ]
    return predicted_classes
  
def draw_box(predicted_classes,image,rect_th= 10,text_size= 3,text_th=3):
    """
    draws box around each object 
    
    predicted_classes: a list where each element contains a tuple that corresponds to information about  the different objects; Each element includes a tuple with the class name, probability of belonging to that class and the coordinates of the bounding box corresponding to the object 
    image : frozen surface 
   
    """

    img=(np.clip(cv2.cvtColor(np.clip(image.numpy().transpose((1, 2, 0)),0,1), cv2.COLOR_RGB2BGR),0,1)*255).astype(np.uint8).copy()
    for predicted_class in predicted_classes:
   
        label=predicted_class[0]
        probability=predicted_class[1]
        box=predicted_class[2]

        cv2.rectangle(img, box[0], box[1],(0, 255, 0), rect_th) # Draw Rectangle with the coordinates
        cv2.putText(img,label, box[0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) 
        cv2.putText(img,label+": "+str(round(probability,2)), box[0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    del(img)
    del(image)
    
#freeing up memory (helper)
def save_RAM(image_=False):
    global image, img, pred
    torch.cuda.empty_cache()
    del(img)
    del(pred)
    if image_:
        image.close()
        del(image)
        
    
#LOADING PRE-TRAINED MODEL
model_ = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model_.eval()

for name, param in model_.named_parameters():
    param.requires_grad = False
print("done")

#calls Faster R-CNN  model  but save RAM
def model(x):
    with torch.no_grad():
        yhat = model_(x)
    return yhat
  
#classes
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
len(COCO_INSTANCE_CATEGORY_NAMES)

#OBJECT LOCALIZATIOn
#Object Localization we locate the presence of objects in an image and indicate the location with a bounding box
img_path='jeff_hinton.png'
half = 0.5
image = Image.open(img_path)

image.resize( [int(half * s) for s in image.size] )

plt.imshow(image)
plt.show()

#Conver the image
transform = transforms.Compose([transforms.ToTensor()])
img = transform(image)

#we can make a prediction,
#The output is a dictionary with several predicted classes, 
#the probability of belonging to that class 
#and the coordinates of the bounding box corresponding to that class.
pred = model([img])
print(pred)

pred[0]['labels'] #the 35 different class predictions, ordered by likelihood scores for potential objects.
#since in this image we only have 1 thing that has been predicted, I can only access pred[0]
#If there were more than one object detected, I would be able to use pred[1] to access the predctions for the second object

pred[0]['scores'] #likelihodd of each class 
#ere we use likelihood as a synonym for probability. 
#Many neural networks output a probability of the output of being a specific class. 
#Here the output is the confidence of prediction, so we use the term likelihood to distinguish between the two

index=pred[0]['labels'][0].item() #class number corresponds to the index of the list with the corresponding  category name
COCO_INSTANCE_CATEGORY_NAMES[index]

#predictions of the box around hte first object
bounding_box=pred[0]['boxes'][0].tolist()
bounding_box

#lets round these values
t,l,r,b=[round(x) for x in bounding_box]

#let us plot this image with the box
img_plot=(np.clip(cv2.cvtColor(np.clip(img.numpy().transpose((1, 2, 0)),0,1), cv2.COLOR_RGB2BGR),0,1)*255).astype(np.uint8)
cv2.rectangle(img_plot,(t,l),(r,b),(0, 255, 0), 10) # Draw Rectangle with the coordinates
pred_class=get_predictions(pred,objects="person")

plt.imshow(cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB))
plt.show()
draw_box(pred_class, img)

#let us delete these two temporary variables again
del img_plot, t, l, r, b
del pred_class

#Let us say we only label if the confidence is more than 0.98
pred_thresh=get_predictions(pred,threshold=0.98,objects="person")
draw_box(pred_thresh,img)
del pred_thresh

#delete these infos to save RAM
save_RAM(image_=True)

#Let us run this on an image with mulitple objects
img_path='DLguys.jpeg'
image = Image.open(img_path)
image.resize([int(half * s) for s in image.size])
plt.imshow(np.array(image))
plt.show()

img = transform(image)
pred = model([img])
pred_thresh=get_predictions(pred,threshold=0.9,)
draw_box(pred_thresh,img,rect_th= 1,text_size= 0.5,text_th=1)
del pred_thresh

#or we can just say what we are looking for:
pred_obj=get_predictions(pred,objects="person") #only looking for persons
draw_box(pred_obj,img,rect_th= 1,text_size= 0.5,text_th=1)
del pred_obj

#we need to set the threshold high enough, or otherwise the model will predict a lot of rectangle boxes to be people or other stuff
pred_thresh=get_predictions(pred,threshold=0.01)
draw_box(pred_thresh,img,rect_th= 1,text_size= 0.5,text_th=1)
del pred_thresh

save_RAM(image_=True)

#OBJECT DETECTION 2
img_path='istockphoto-187786732-612x612.jpeg'
image = Image.open(img_path)
image.resize( [int(half * s) for s in image.size] )
plt.imshow(np.array(image))
plt.show()
del img_path

#Let us detect all the images
img = transform(image) #conver to rgb and resize
pred = model([img]) #use our model to predict
pred_thresh=get_predictions(pred,threshold=0.97) #only things which we are confident more than 0.97
draw_box(pred_thresh,img,rect_th= 1,text_size= 1,text_th=1)
del pred_thresh

save_RAM(image_=True)

#Let us say, we are only interested in cats and dogs
img = transform(image)
pred = model([img])
pred_obj=get_predictions(pred,objects=["dog","cat"])
draw_box(pred_obj,img,rect_th= 1,text_size= 0.5,text_th=1)
del pred_obj

#we set the threshold too low, we may detect objects with a low likelihood of being correct; 
#here, we set the threshold to 0.7, and we incorrectly detect a cat
img = transform(image)
pred = model([img])
pred_thresh=get_predictions(pred,threshold=0.70,objects=["dog","cat"])
draw_box(pred_thresh,img,rect_th= 1,text_size= 1,text_th=1)
del pred_thresh
save_RAM(image_=True)

#The classifer can also detect other objects like cars
img_path='watts_photos2758112663727581126637_b5d4d192d4_b.jpeg'
image = Image.open(img_path)
image.resize( [int(half * s) for s in image.size] )
plt.imshow(np.array(image))
plt.show()
del img_path

img = transform(image)
pred = model([img])
pred_thresh=get_predictions(pred,threshold=0.997)
draw_box(pred_thresh,img)
del pred_thresh

#Or upload any image you like!
url='https://www.plastform.ca/wp-content/themes/plastform/images/slider-image-2.jpg'

#transform to RGB
image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
del url
img = transform(image )
pred = model([img])
pred_thresh=get_predictions(pred,threshold=0.95)
draw_box(pred_thresh, img)
del pred_thresh

save_RAM(image_=True)

