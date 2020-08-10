# Dog Identification App

### Table of Contents
1. [Project Definition](#Project-Definition)
2. [Installation](#Installation)
3. [File Descriptions](#File-Descriptions)
4. [Analysis](#Analysis)
5. [Methodology](#Methodology)
6. [Results](#Results)
7. [Web App Instructions](#Web-App-Instructions)
8. [Conclusion](#Conclusion)
9. [Acknowledgements](#Acknowledgements)

### Project Definition
#### Project Motivation
This project is being completed to fulfil the Capstone project requirement for Udacity's Data Science Nanodegree program. The objective of this project is to develop an algorithm using Convolutional Neural Networks (CNN) and Deep Learning for computer image recognition which can be used as part of a web app.

#### Project Overview 
This project develops an algorithm using Convolutional Neural Networks that is used as part of a web app.  The algorithm is composed several different models for dog detection in images, human detection in images, and dog breed prediction.  Given an image of a dog, the algorithm will identify an estimate of the dog’s breed.  If supplied an image of a human, the algorithm will identify the resembling dog breed.  

#### Problem Statement
There are several problems which need to be solved for this project.  The first problem which needs to be solved is the problem of being able to correctly identify a dog from an image.  As a solution to this problem, a ResNet-50 model pre-trained on the [ImageNet](http://image-net.org) dataset is used to detect dogs in images.  The ResNet architecture was introduced by He et al. in their 2015 paper entitled ["Deep Residual Learning for Image Recognition." ](https://arxiv.org/abs/1512.03385)

The next problem which needs to be solved is the problem of being able to identify humans in an image.  As a solution to this problem, OpenCV’s implementation of Haar feature-based cascade classifiers is used as a pre-trained face detector to detect human faces in images. The Haar Feature-based Cascade Classifier for Object Detection was initially proposed by Paul Viola and Michael Jones in their 2001 paper entitled ["Rapid Object Detection using a Boosted Cascade of Simple Features"](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf) and it was improved upon by Rainer Lienhart and Jochen Maydt in their 2002 paper entitled ["An Extended Set of Haar-like Features for Rapid Object Detection."](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.214.9150&rep=rep1&type=pdf)  The human face detector requires the reasonable expectation that the human images provided as input have a clear view of the face.  To detect human faces in images where the face is not clearly presented, then a model would have to be trained on a dataset containing images where the face is covered, obscured, in profile, or only partially shown, so that the model would know how to detect humans from images where the face is not clearly presented.

The last problem which needs to be solved for this project is the problem of being able to predict a dog’s breed based on an image.  As a solution to this problem, several CNN models are tested for image classification to predict a dog’s breed including a full CNN model built from scratch, along with CNNs created with Transfer Learning using the VGG-16 and Inception bottleneck features.  The VGG architecture was introduced by Simonyan and Zisserman in their 2014 paper entitled ["Very Deep Convolutional Networks for Large Scale Image Recognition"](https://arxiv.org/abs/1409.1556) and the Inception CNN architecture was first introduced by Szegedy et al. in their 2014 paper entitled ["Going Deeper with Convolutions."](https://arxiv.org/abs/1409.4842)  In the end, the CNN creating with Transfer Learning using the InceptionV3 bottleneck features had the highest test accuracy and was implemented in the final developed algorithm.

#### Metrics
The performance of the dog detection model will be measured by the number of dogs detected by the model from images containing dogs, and the number of dogs detected by the model from images not containing dogs.  

Similarly, the performance of the human detection model will be measured by the number of human faces detected by the model from images containing humans, and the number of humans detected by the model from images not containing humans.

The metric used to measure the performance of the dog breed prediction model will be the accuracy of correctly classifying the dog's breed from an image based on the test set of dog image data.  Also, there is some imbalance in the distribution of images per dog breed category in the training dataset as seen in Figure-01 under the Data Visualization section.  Therefore, scikit-learn's weighted averaging for multiclass targets of precision, recall, and F1-score will also be used as performance metrics to account for the imbalance as described in the [scikit-learn documentation.](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)

### Installation
This project uses Python version 3.7.6 and the following libraries which should be installed: 
* h5py==2.10.0
* ipykernel==5.1.4
* keras==2.3.1
* matplotlib==3.1.3
* numpy==1.19.0
* opencv-python==4.3.0.36
* pandas==1.0.1
* pillow==7.0.0
* scikit-learn==0.22.1
* scipy==1.4.1
* tensorflow==2.2.0
* tqdm==4.42.1

Additionally, the web-app uses the following libraries:
* flask==1.1.1
* plotly==4.9.0
* pickle

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

### Analysis
#### Data Exploration 
There are 8351 dog images in the dog dataset with 133 total dog categories.  6680 dog images are used for training, 835 dog images are used for validation, and 836 dog images are used for testing. As can be seen in Figure-02 below, some abnormalities do appear in the dog image dataset including images containing multiple dogs of different breeds, and images containing both dogs and humans. 

There are 13233 human images in the human dataset.  There do not appear to be any abnormalities in the dataset, and as can be seen in Figure-03 below, there appears to be a diverse collection of images within the human image dataset.  

#### Data Visualization 
Figure-01 below shows the number of dog images in the 133 dog breed categories in the training dataset:
<p align="center"><b>Figure-01: Distribution of Dog Images per Breed Category in the Training Dataset</b></p>

![Figure1](https://raw.githubusercontent.com/dujamaa/Proj4_DogBreeds/master/images/training_dogbreed_distribution.png)

Figure-02 below shows the first 36 dog images in the training dataset:
<p align="center"><b>Figure-02: First 36 Dog Images in the Training Dataset</b></p>

![Figure2](https://raw.githubusercontent.com/dujamaa/Proj4_DogBreeds/master/images/training_sample_dog_images.png)

Figure-03 below shows the first 36 human faces in the human image dataset:
<p align="center"><b>Figure-03: First 36 Images in the Human Image Dataset</b></p>

![Figure3](https://raw.githubusercontent.com/dujamaa/Proj4_DogBreeds/master/images/sample_human_images.png)

### Methodology
#### Data Preprocessing
When using TensorFlow backend, Keras CNNs require a 4D array (or 4D tensor) as input.  An image is resized to a square image of size 224 X 224 pixels, and then converted to an array, which is then resized to a 4D tensor.  For a color image, the returned tensor has shape (1, 224, 224, 3).  Additionally, the 4D tensor is prepared for Keras CNNs by converting the RGB image to BGR by reordering the channels.  An additional normalization step is included for all pre-trained models that the mean pixel must be subtracted from every pixel in each image.  Once these preprocessing steps are done, the ResNet-50 model can be used to make predictions as part of the dog detector algorithm.  Data is pre-processed for the CNN to classify dog breeds by rescaling the images by dividing every pixel in every image by 255.

#### Implementation & Refinement
The model which was ultimately implemented in the final algorithm for image classification to predict a dog’s breed was a CNN created with Transfer Learning using the bottleneck features from the pre-trained InceptionV3 model as a fixed feature extractor.  Since the dog image dataset is relatively small in size and similar to the ImageNet data, then to use transfer learning, the last fully connected layer of the pre-trained InceptionV3 model should be removed and replaced with a layer containing the 133 dog breed classes in the dog image data set. This is accomplished in my model by feeding the output of the of the pre-trained InceptionV3 model as input into a Global Average Pooling (GAP) layer for extreme dimensionality reduction, and adding a fully connected dense layer with a Softmax activation function to return probability estimates for the 133 dog breeds.  To refine the model, the weights are saved while the model is training, and the model with the best validation loss is loaded and used for predictions.  There were no complications or difficulties during the coding process for this model as it was straightforward as seen in the code block below:
```
    ### Obtain bottleneck features from another pre-trained CNN.
    bottleneck_features = np.load('bottleneck_features/DogInceptionV3Data.npz')
    train_InceptionV3 = bottleneck_features['train']
    valid_InceptionV3 = bottleneck_features['valid']
    test_InceptionV3 = bottleneck_features['test']

    ### Define model architecture:
    InceptionV3_model = Sequential()
    InceptionV3_model.add(GlobalAveragePooling2D(input_shape=train_InceptionV3.shape[1:]))
    InceptionV3_model.add(Dense(133, activation='softmax'))
    InceptionV3_model.summary()
    
    ### Compile the model:
    InceptionV3_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    ### Train the model:
    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.InceptionV3.hdf5', verbose=1, save_best_only=True)
    
    InceptionV3_model.fit(train_InceptionV3, train_targets, validation_data=(valid_InceptionV3, valid_targets), 
                          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)
			  
    ### Load the model weights with the best validation loss.
    InceptionV3_model.load_weights('saved_models/weights.best.InceptionV3.hdf5')
```
### Results
#### Model Evaluation and Validation
The final algorithm combined OpenCV’s implementation of Haar feature-based cascade classifiers to detect human faces in images, the ResNet-50 model to detect dogs in images, and a CNN creating with Transfer Learning using the InceptionV3 bottleneck features for image classification to predict a dog’s breed.  Below are the results of the metrics used to evaluate the model's qualities:
* In assessing the human face detector, the model was able to detect a human face in 99% of 100 human images, and it detected a human face in 12% of 100 dog images.
* In assessing the ResNet-50 model dog detector, the model was able to detect a dog in 100% of 100 dog images, and it detected a dog in 1% of 100 human images.
* For the CNN developed from scratch to predict a dog’s breed, the model attained 2.0335% test accuracy in 5 epochs, with Precision = 0.3033%, Recall = 2.0335%, and F1 score = 0.4625%.
* For the CNN created with Transfer Learning using the VGG-16 bottleneck features, the model attained 72.8469% test accuracy in 20 epochs, with Precision = 75.4096%, Recall = 72.8469%, and F1 score = 72.0778%.
* For the CNN created with Transfer Learning using the InceptionV3 bottleneck features, which was ultimately used in the final algorithm, the model attained 80.1435% test accuracy in 20 epochs, with Precision = 83.7576%, Recall = 80.1435%, and F1 score = 79.8980%.
* The CNN created with Transfer Learning using the InceptionV3 bottleneck features was ultimately used in the final algorithm.  To validate the robustness of the model’s solution, k-fold cross validation was performed on 10 validation folds.  The validation performance of the model was relatively stable across the 10 validation folds, with the average cross validation score being 80.70% +/- 1.89%, with a minimum cross validation score of 77.69% and a maximum cross validation score of 84.43%.  Therefore the model appears to be robust against small perturbations in the training data.

#### Justification
At the beginning of the project I experimented with using the VGG19, ResNet50, and InceptionV3 bottleneck features to build the CNN with Transfer Learning to classify dog breeds.  The VGG19 bottleneck features provided the lowest test accuracy, and the test accuracy of the ResNet50 bottleneck features was inconsistent when testing the model on my local computer and on a GPU.  In the end, the Inception bottleneck features provided the best and most consistent test accuracy.  Since the InceptionV3 bottleneck features seemed to work best, the final model was improved by using the Inception V3 bottleneck features. 

The output of the algorithm performed better than expected. The algorithm was tested on six images: 2 human images, 2 dog images, 1 cat image, and 1 wolf image. The algorithm correctly identified the humans in both images and returned a resembling dog breed. The algorithm correctly identified that neither a dog nor a human was in the cat image. In an attempt to trick the algorithm, an image of a wolf was provided as input, but the algorithm correctly identified that neither a dog nor a human was in the wolf image. The algorithm correctly identified the dog breed for both dog images provided as input. Although one dog image provided as input was named “Pitbull,” the algorithm predicted the dog breed as “American Staffordshire Terrier,” and both American Staffordshire Terriers and American Pitbull Terriers look very similar and are both considered “Pitbull” type dogs.

### Web App Instructions
1. From a terminal, Run the following command in the `path-to-project/app` directory of the repo to run the web app locally: `python dog.py`

2. Go to http://0.0.0.0:3001/ (on Windows go to http://localhost:3001/) to view the web app
<p align="center"><b>Figure-04: Initial Screen of the web app</b></p>

![Figure4](https://raw.githubusercontent.com/dujamaa/Proj4_DogBreeds/master/images/screenshots/01.png)

3. When the web app opens, press the **Browse** button to select an image, and the image will appear in the app as seen in the screen below:
<p align="center"><b>Figure-05: Image loaded into web app after pressing the Browse button and selecting an image</b></p>

![Figure5](https://raw.githubusercontent.com/dujamaa/Proj4_DogBreeds/master/images/screenshots/02.png)

4. After selecting an image, press the **Classify Image** button to get the result of the web app.  After waiting for a moment, a result is returned to the right of the image. If an image of a dog is provided as input, then the predicted breed of the dog is returned as seen in the screen below: 
<p align="center"><b>Figure-06: After pressing Classify Image, the web app returns a dog's predicted breed if a dog image is provided</b></p>

![Figure6](https://raw.githubusercontent.com/dujamaa/Proj4_DogBreeds/master/images/screenshots/03.png)

5. If an image of a human is provided as input, then the resembling dog breed is returned to the right of the image as seen in the screen below:
<p align="center"><b>Figure-07: After pressing Classify Image, the web app returns a resembling dog breed if a human image is provided</b></p>

![Figure7](https://raw.githubusercontent.com/dujamaa/Proj4_DogBreeds/master/images/screenshots/04.png)

6. If an image is provided as input which contains neither a dog nor a human, then a message is returned to the right of the image stating that No Dog or Human was detected in the image, as seen in the screen below:
<p align="center"><b>Figure-08: If an image containing no dog or human is provided as input, then an error message is returned</b></p>

![Figure8](https://raw.githubusercontent.com/dujamaa/Proj4_DogBreeds/master/images/screenshots/05.png)

### Conclusion
#### Reflection
The objective of this project was to develop an algorithm which could detect the presence or absence of dogs or humans in an image, and predict the dog’s breed based on the image of a dog, or find a dog breed that resembles the image of a human provided as input to the algorithm.  In order to develop this algorithm several problems had to be solved including detecting dogs in images, detecting humans in images, and predicting a dog’s breed based on an image.  As an end-to-end summary of the solution to the problem, the algorithm was built by combining three different models to solve each problem: a ResNet-50 model was used to detect dogs in images, OpenCV’s implementation of Haar feature-based cascade classifiers was used to detect human faces in images, and a CNN created with transfer learning using the InceptionV3 bottleneck features was used to predict a dog’s breed from an image.  The algorithm was implemented in an app that can be ran locally.  

Although it was not required, one aspect of this project that was difficult was deploying the app to the web.  I attempted to deploy the app to the web using [Heroku]( https://www.heroku.com/), however Heroku would not allow installation of the version of tensorflow (2.2.0) that was being used for this project, and rolling back the version of tensorflow to a version allowed by Heroku would cause the app to not run.  One thing about this project I found it interesting and fascinating is how well the algorithm was able to distinguish images of dogs from images of wolves.  Another interesting part of this project was the features in human images that the algorithm used to find resembling dog breeds.

#### Improvement
Three possible points of improvement for my algorithm could be 1) including dropout layers to minimize overfitting, 2) increasing the number of epochs or adjusting the learning rate, and 3) using image augmentation in the training dataset to account for scale, rotation, and translation invariance to increase the accuracy of the system.  Currently, the algorithm does not take too much time to train and the entire jupyter notebook can be run in a reasonable amount of time.  If the number of epochs are increased or images in the training dataset are augmented, then it may improve the algorithm, but it will also likely increase the processing time.

### Acknowledgements
* [Udacity](https://www.udacity.com/) is acknowledged for this [dog project](https://github.com/udacity/dog-project).
