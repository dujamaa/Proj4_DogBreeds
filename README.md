# Dog Identification App

### Table of Contents
1. [Project Motivation](#Project-Motivation)
2. [Project Overview](#Project-Overview)
2. [Installation](#Installation)
3. [File Descriptions](#File-Descriptions)
4. [Problem  Statement](#Problem-Statement)
5. [Analysis](#Analysis)
6. [Results](#Results)
7. [Web App Instructions](#Web-App-Instructions)
8. [Conclusion](#Conclusion)
9. [Acknowledgements](#Acknowledgements)


[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Keras Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"


### Project Motivation
This project is being completed to fulfil the Capstone project requirement for Udacity's Data Science Nanodegree program. The objective of this project is to develop an algorithm using Convolutional Neural Networks (CNN) and Deep Learning for computer image recognition which can be used as part of a web app.

### Project Overview 
This project develops an algorithm using Convolutional Neural Networks that is used as part of a web app.  The algorithm is composed several different models for dog detection in images, human detection in images, and dog breed prediction.  Given an image of a dog, the algorithm will identify an estimate of the dog’s breed.  If supplied an image of a human, the algorithm will identify the resembling dog breed.  


### Installation
This project uses Python version 3.7+ and the following libraries which should be installed: 
* pandas 
* numpy
* sklearn
* scipy
* keras
* tensorflow
* matplotlib
* tqdm 
* opencv
* cv2
* pillow
* ipykernel
* h5py
* glob

Additionally, the web-app uses the following libraries:
* pickle
* flask
* plotly

After downloading the repository, download the following datasets and bottleneck features:
1. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`. 
2. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.  
3. Download the [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) for the dog dataset.  Place it in the repo, at location `path/to/dog-project/bottleneck_features`.
4. Download the [Inception bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz) for the dog dataset.  Place it in the repo, at location `path/to/dog-project/bottleneck_features`.

### File Descriptions
Below are the descriptions for the main files and folders of this project:
* app - folder containing files for the web app 
	* templates - folder containing html code for the web app 
		* index.html - main page of web app
	* models - folder containing models for the web app
		* InceptionV3_model.pkl – pickled InceptionV3 model
		* dog_names.pkl – pickled list of dog names
		* haarcascade_frontalface_alt.xml – pre-trained face detector
	* dog.py - python flask file that runs web app
	* extract_bottleneck_features.py – python code to extract bottleneck features
* bottleneck_features – folder for saving downloaded bottleneck features
* haarcascades – folder containing pre-trained face detector
	* haarcascade_frontalface_alt.xml – pre-trained face detector
* images – folder containing images of dogs, humans, and other images for the project
* saved_models – folder where the best model weights are saved 
	* weights.best.InceptionV3.hdf5 – best saved model weights from Inception model
* dog_app.ipynb – main Jupyter notebook containing python code for this project.
* extract_bottleneck_features.py – python code to extract bottleneck features

### Problem Statement
There are several problems which need to be solved for this project.  First, there is the problem of being able to correctly identify a dog from an image.  Next, there is the problem of being able to identify humans in an image.  Lastly, there is the problem of being able to predict a dog’s breed based on an image.

OpenCV’s implementation of Haar feature-based cascade classifiers is used as a pre-trained face detector to detect human faces in images.  The human face detector requires the reasonable expectation that the human images provided as input have a clear view of the face.  To detect human faces in images where the face is not clearly presented, then a model would have to be trained on a dataset containing images where the face is covered, obscured, in profile, or only partially shown, so that the model would know how to detect humans from images where the face is not clearly presented.

A ResNet-50 model pre-trained on the ImageNet dataset is used to detect dogs in images. 

Several CNN models are tested for image classification to predict a dog’s breed including a full CNN model built from scratch, along with CNNs created with Transfer Learning using the VGG-16 and Inception bottleneck features.  In the end, the CNN creating with Transfer Learning using the Inception bottleneck features had the highest test accuracy and was implemented in the final developed algorithm.

### Analysis
#### Data Exploration 
There are 8351 dog images in the dog dataset with 133 total dog categories.  6680 dog images are used for training, 835 dog images are used for validation, and 836 dog images are used for testing. There are 13233 human images in the human dataset.

#### Data Preprocessing
When using TensorFlow backend, Keras CNNs require a 4D array (or 4D tensor) as input.  An image is resized to a square image of size 224 X 224 pixels, and then converted to an array, which is then resized to a 4D tensor.  For a color image, the returned tensor has shape (1, 224, 224, 3).  Additionally, the 4D tensor is prepared for Keras CNNs by converting the RGB image to BGR by reordering the channels.  An additional normalization step is included for all pre-trained models that the mean pixel must be subtracted from every pixel in each image.  Once these preprocessing steps are done, the ResNet-50 model can be used to make predictions as part of the dog detector algorithm.  Data is pre-processed for the CNN to classify dog breeds by rescaling the images by dividing every pixel in every image by 255.

#### Implementation & Refinement
The model which was ultimately used in the final algorithm for image classification to predict a dog’s breed was a CNN created with Transfer Learning using the bottleneck features from the pre-trained InceptionV3 model as a fixed feature extractor.  Since the dog image dataset is relatively small in size and similar to the ImageNet data, then to use transfer learning, the last fully connected layer of the pre-trained InceptionV3 model should be removed and replaced with a layer containing the 133 dog breed classes in the dog image data set. This is accomplished in my model by feeding the output of the of the pre-trained InceptionV3 model as input into a Global Average Pooling (GAP) layer for extreme dimensionality reduction, and adding a fully connected dense layer with a Softmax activation function to return probability estimates for the 133 dog breeds.  To improve upon the model, the weights are saved while the model is training, and the model with the best validation loss is loaded and used for predictions.

### Results
* In assessing the human face detector, the model was able to detect a human face in 99% of 100 human images, and it detected a human face in 12% of 100 dog images.
* In assessing the ResNet-50 model dog detector, the model was able to detect a dog in 100% of 100 dog images, and it detected a do in 1% of 100 human images.
* For the CNN developed from scratch to predict a dog’s breed, the model attained 2.6% test accuracy in 5 epochs.
* For the CNN created with Transfer Learning using the VGG-16 bottleneck features, the model attained 72.97% test accuracy in 20 epochs.
* For the CNN created with Transfer Learning using the Inception bottleneck features, which was ultimately used in the final algorithm, the model attained 80.3% test accuracy in 20 epochs.

### Web App Instructions
1. From a terminal, Run the following command in the path-to-project/app directory of the repo to run the web app locally: python dog.py
2. Go to http://0.0.0.0:3001/ (on Windows go to http://localhost:3001/)
3. When the web app opens, press the browse button to select an image.
4. After selecting an image, press the Classify image button to get the result of the web app.
5. After waiting for the web app to process the image, a result is returned which either predicts the dog's breed if an image of a dog was provided as input, or states the dog breed that resembles the human if a human image was provided as input, or outputs a message that neither a dog nor a human was in the image provided as input.

### Conclusion
The output of my algorithm is better than I expected. I tested the algorithm on six images: 2 human images, 2 dog images, 1 cat image, and 1 wolf image. My algorithm correctly identified the humans in both images and returned a resembling dog breed. My algorithm correctly identified that neither a dog nor a human was in the cat image. I thought I would be able to trick the algorithm by inputting an image of a wolf, but my algorithm correctly identified that neither a dog nor a human was in the wolf image. My algorithm correctly identified the dog breed for both dog images I provided as input. Although one dog image I provided as input was named “Pitbull,” my algorithm predicted the dog breed as “American Staffordshire Terrier,” and after doing some google searching, I learned that both American Staffordshire Terriers and American Pitbull Terriers look very similar and are both considered “Pitbull” type dogs. Three possible points of improvement for my algorithm could be 1) including dropout layers to minimize overfitting, 2) increasing the number of epochs or adjusting the learning rate, and 3) using image augmentation in the training dataset to account for scale, rotation, and translation invariance to increase the accuracy of the system.

### Acknowledgements
* [Udacity](https://www.udacity.com/) is acknowledged for this [dog project](https://github.com/udacity/dog-project).
