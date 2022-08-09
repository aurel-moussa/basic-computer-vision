#We will will train a deep neural network for image classification using transfer learning
#the image dataset will automatically be download from the IBM CV Studio account. 
#We will experiment with different hyperparameters.

#We will train am image classifier using CV Studio (easy and collaborative open source image annotation tool for teams and individuals) to 
#1) annotate images
#2) build an external webapp

#The whole model training will be done within a Juypter notebook

#In practice, very few people train an entire Convolutional Network from scratch (with random initialization), 
#because it is relatively rare to have a dataset of sufficient size 
#Instead, it is common to pretrain a ConvNet on a very large dataset in the lab, then use this Network to train your model
#We will use the Convolutional Network as a feature generator, only training the output layer  
#In general, 100-200 images will give you a good starting point, and it only takes about half an hour  
#Usually, the more images you add, the better your results, but it takes longer and the rate of improvement will have decreasing marginal rate of return


