
import plotly
import pandas as pd
import cv2 
import os
import numpy as np            
import matplotlib.pyplot as plt                        
#%matplotlib inline 

from glob import glob
from keras.utils import np_utils
from sklearn.externals import joblib
from sklearn.datasets import load_files     
from flask import Flask, render_template, request, jsonify

from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image                  
from tqdm import tqdm

from extract_bottleneck_features import *

#import json
#from nltk.stem import WordNetLemmatizer
#from nltk.tokenize import word_tokenize
#from plotly.graph_objs import Bar
#from sqlalchemy import create_engine

app = Flask(__name__)

def face_detector(img_path):
    
    # extract pre-trained face detector
    face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_alt.xml')
    
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

import keras.backend.tensorflow_backend as tb #added to workaround error

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    tb._SYMBOLIC_SCOPE.value = True #added to workaround error
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

def InceptionV3_predict_breed(img_path):
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