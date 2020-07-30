
import plotly
import pandas as pd
import cv2 
import os
import numpy as np            
import matplotlib.pyplot as plt                        

from glob import glob
from keras.utils import np_utils
from sklearn.externals import joblib
from sklearn.datasets import load_files     
from flask import Flask, render_template, request, jsonify

from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image                  
from tqdm import tqdm

from extract_bottleneck_features import *


app = Flask(__name__)

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_alt.xml')

def face_detector(img_path):
    """Detect if a human face is in an image
    Args: 
        img_path: file path and name of image to detect                     
    Returns: 
        "True" if human face is detected in image stored at img_path, 
        "False" otherwise
    """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

def path_to_tensor(img_path):
    """Pre-process image data by taking a string-valued file path 
    to a color image as input and returning a 4D tensor suitable 
    for supplying to a Keras CNN
    Args:
        img_path: file path and name of image                     
    Returns: 
        4D array (4D tensor) of shape (nb_samples=1,rows=224,columns=224,channels=3)
    """
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    """Pre-process image data by taking a numpy array of 
    string-valued image paths as input and returning a 4D tensor
    Args:
        img_paths: numpy array of string-valued image paths                     
    Returns: 
        4D array (4D tensor) of shape (nb_samples,rows=224,columns=224,channels=3)
        where nb_samples is the number of samples (images) in the array of image paths
    """
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

import keras.backend.tensorflow_backend as tb #added to workaround error

def ResNet50_predict_labels(img_path):
    """Takes image and returns prediction vector
    Args:
        img_path: file path and name of image                     
    Returns: 
        prediction vector for image located at img_path
    """
    # returns prediction vector for image located at img_path
    tb._SYMBOLIC_SCOPE.value = True #added to workaround error
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def dog_detector(img_path):
    """Detect if a dog is in an image
    Args: 
        img_path: file path and name of image to detect                     
    Returns: 
        "True" if dog is detected in image stored at img_path, 
        "False" otherwise
    """
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

def InceptionV3_predict_breed(img_path):
    """Predict dog breed from an image using pre-train InceptionV3 model 
    Args:
        img_path: file path and name of image                     
    Returns: 
        Predicted dog breed using InceptionV3 model
    """
    # extract bottleneck features
    bottleneck_feature = extract_InceptionV3(path_to_tensor(img_path))
    
    # load model and dog_names
    from sklearn.externals import joblib
    InceptionV3_model = joblib.load("models/InceptionV3_model.pkl")
    dog_names = joblib.load("models/dog_names.pkl")

    # obtain predicted vector
    predicted_vector = InceptionV3_model.predict(bottleneck_feature)
    
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]

def Dog_breed_algorithm(img_path):
    """ Takes a filepath to an image and determines if a dog or a human 
        is in the image. If a dog is detected in the image, then the 
        predicted breed is returned. If a human is detected in the image, 
        then the resembling dog breed is returned. If neither dog nor 
        human is dected in the image, then a error message is returned.
    Args:
        img_path: file path and name of image                     
    Returns: 
        detect_message: a text string stating if dog, human, or neither
        was detected in the image.
        Predicted_Breed: the predicted breed of the dog or resembling 
        breed of the human
    """    
    # display image
    #img = cv2.imread(img_path)
    #cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #plt.imshow(cv_rgb)
    #plt.show()
    
    # if a dog is detected in image, return the predicted breed
    if dog_detector(img_path):
        detect_message = "DOG detected. The predicted breed is ..."       
        Predicted_Breed = InceptionV3_predict_breed(img_path)
        
    # if a human is detected in the image, return the resembling dog breed
    if face_detector(img_path):
        detect_message = "HUMAN detected. The resembling Dog breed is ..."         
        Predicted_Breed = InceptionV3_predict_breed(img_path)
        
    # if neither a dog nor a human is in the image, return output that indicates an error
    if (dog_detector(img_path) == False) and (face_detector(img_path) == False):
        detect_message = "No Dog or Human was detected in the image!"
        Predicted_Breed = ""
    return detect_message, Predicted_Breed


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():   
    # render web page with plotly graphs
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files["file"]
        file.save(os.path.join("static/img", file.filename))
        name_of_file = "static/img/" + file.filename
        detect_message, Predicted_Breed = Dog_breed_algorithm(name_of_file)
    return render_template('index.html', 
                           name_of_file = name_of_file, 
                           detect_message=detect_message, 
                           Predicted_Breed=Predicted_Breed )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()