# Load libraries and packages needed for image classification project
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers
from IPython.display import HTML
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "./Images")

breeds = dataset.class_names
print(breeds)

# Set the arguments for the tensorflow instance
args = {
    'labels':'inferred',          # Infers the name of the dog by the name of the directory the picture is in
    'label_mode':'categorical',   # Each breed is one category
    'batch_size': 10,             # how many images are loaded and processed at once by neural network
    'image_size': (256,256),      # resize all images to the same size 
    'seed': 1,                    # set seed for reproducability
    'validation_split': .2,       # split training and testing : 80% train and 20% test
    'class_names': breeds         # name of the categories
}

train = tf.keras.utils.image_dataset_from_directory(           # Loads images from directory into tensorflow training dataset
    "./Images",
    subset='training',
    **args
)

test = tf.keras.utils.image_dataset_from_directory(           # Loads images from directory into tensorflow  testing dataset
    "./Images",
    subset='validation',
    **args
)

train = train.cache().prefetch(buffer_size=tf.data.AUTOTUNE)   # Caches pictures in memory rather than hard disk to make algorithm more efficient
test = test.cache().prefetch(buffer_size=tf.data.AUTOTUNE)   # Caches pictures in memory rather than hard disk to make algorithm more efficient

# import keras packages for modeling and build sequential model

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

model = Sequential ([
    layers.Rescaling(1./255),    # Pixels in numpy array are from 0 - 255, so we rescale the pixels into numbers between 0-1 in order to help neural network be more efficient
    layers.Conv2D(16,3,padding='same',activation='relu',input_shape=(255,255,3)),             # Create a convulutional layer that scans images and generates new matrices with features from the images, will do this 16 times, looking at 3x3 pixels nat a time(window)
    layers.Flatten(),
    layers.Dense(128,activation='relu'),  # dense network will take flattened layer and help facilitate predictions
    layers.Dense(len(breeds))             # this line will make the prediction
])

# Compile the model

model.compile(optimizer='RMSprop',loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),metrics =['accuracy'])     # optimizer tells model how to predict error and how to iterate, and loss function calculates error

# fit the model

history = model.fit(
    train,
    validation_data=test,
    epochs=7,
    verbose=1
)

history_df = pd.DataFrame.from_dict(history.history)
history_df[['accuracy','val_accuracy']].plot();
# Account for overfitting and increas emodel accuracy
# paste model from before into function so as to not have to change the code every time

def train_model(network,epochs=5):
    model = Sequential(network)
    model.compile(optimizer='RMSprop',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics =['accuracy'])
    history = model.fit(
    train,
    validation_data=test,
    epochs=7,
    verbose=1
)
    history_df = pd.DataFrame.from_dict(history.history)
    return history_df,model

# same netwrok as before, modified with new layers

network = [
    layers.Rescaling(1./255),    
    layers.Conv2D(16,4,padding='same',activation='relu',input_shape=(256,256,3)),     # increase window size to 4 from 3
    layers.MaxPooling2D(),                                                            # add max pooling 2d layer to reduce overfit and reduce number of parameters
    layers.Conv2D(32,4,padding='same',activation='relu',input_shape=(256,256,3)),     # add second convolutional layer with increased filters to 32, to let network pick up higher level features
    layers.MaxPooling2D(),                                                           # add another max pooling layer
    layers.Conv2D(64,4,padding='same',activation='relu',input_shape=(256,256,3)),    # add another convolutional layer with 64 filters for even higher level features
    layers.MaxPooling2D(),                                                           # another max pooling layer
    layers.Dropout(.2),                                                              # Dropout layer helps with overfitting by setting some outputs to 0 randomly, so network doesnt become too linked to trainjing data
    layers.Flatten(),
    layers.Dense(128,activation='relu'),  
    layers.Dense(len(breeds))             
]

# run model again

history_df,model = train_model(network)
history_df[['accuracy','val_accuracy']].plot();  # not much of an accuracy increase: increase epochs and see changes

# Still overfitting

# Data augmentaion may help with overfitting with keras layers, set to a sequential layer
# set seeds for reproducibility 

data_augmentaion = tf.keras.Sequential([
    layers.RandomFlip('horizontal',seed=1),         # Randomly flips images from left to right which the model wuill see as a new image and increase accuracy
    layers.RandomRotation(.2,seed=1),               # Randomly raotates images for more information
    layers.RandomZoom(.2,seed=1)                   #randomly zooms images for more information for the model 
])

full_network = [data_augmentaion] + network

history_df, model = train_model(full_network)
preds = model.predict(test) 
import numpy as np 
predicted_class = np.argmax(preds,axis=1) 
print(predicted_class)
actual_labels = np.concatenate([y for x,y in test],axis=0)   # flattens out batches and pulls out labels
print(actual_labels)
actual_class = np.argmax(actual_labels,axis=1)

import itertools
from PIL import Image

actual_image = [x.numpy().astype('uint8') for x,y in test]
actual_image = list(itertools.chain.from_iterable(actual_image))
actual_image = [Image.fromarray(a) for a in actual_image]
# create datframe from predicted, actual, and the images of the dogs

pred_df = pd.DataFrame(zip(predicted_class,actual_class,actual_image),columns=['prediction','actual','image'])
pred_df['prediction'] = pred_df['prediction'].apply(lambda x : breeds[x])
pred_df['actual'] = pred_df['actual'].apply(lambda x : breeds[x])

# Render actal images instead of image data

import base64
import io
def image(img):
    with io.BytesIO() as buffer:
        img.save(buffer,'png')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f'<img src="data:image/jpeg;base64,{img_str}">'
  
    
# this is all a little convoluted and had to look up how to do all of this, working with images and HTML
    # Look at predictions and acrtual dogs

  
pred_df.head(200).style.format({'image':image})

# Labs and dobermans seem hard for the model to predict

