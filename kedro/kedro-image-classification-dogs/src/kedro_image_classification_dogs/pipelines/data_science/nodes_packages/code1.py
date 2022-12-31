from typing import Tuple

import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pickle


def premier_modele():
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "data/01_raw/Images")

    # set the breed of dogs that the program will be classifying
    breeds = dataset.class_names

    # Set the arguments for the tensorflow instance
    args = {
        'labels': 'inferred',  # Infers the name of the dog by the name of the directory the picture is in
        'label_mode': 'categorical',  # Each breed is one category
        'batch_size': 32,  # how many images are loaded and processed at once by neural network
        'image_size': (224, 224),  # resize all images to the same size
        'seed': 1,  # set seed for reproducability
        'validation_split': .2,  # split training and testing : 80% train and 20% test
        'class_names': breeds  # name of the categories
    }

    # Training data

    train = tf.keras.utils.image_dataset_from_directory(  # Loads images from directory into tensorflow training dataset
        "data/01_raw/Images",
        subset='training',
        **args
    )

    # Test Data

    test = tf.keras.utils.image_dataset_from_directory(  # Loads images from directory into tensorflow  testing dataset
        "data/01_raw/Images",
        subset='validation',
        **args
    )
    train = train.cache().prefetch(
        buffer_size=tf.data.AUTOTUNE)  # Caches pictures in memory rather than hard disk to make algorithm more
    # efficient
    test = test.cache().prefetch(
        buffer_size=tf.data.AUTOTUNE)  # Caches pictures in memory rather than hard disk to make algorithm more
    # efficient

    model = Sequential ([
    layers.Rescaling(1./255),    # Pixels in numpy array are from 0 - 255, so we rescale the pixels into numbers between 0-1 in order to help neural network be more efficient
    layers.Conv2D(16,3,padding='same',activation='relu',input_shape=(224,224,3)), # Create a convulutional layer that scans images and generates new matrices with features from the images, will do this 16 times, looking at 3x3 pixels nat a time(window)
    layers.Flatten(),
    layers.Dense(128,activation='relu'),  # dense network will take flattened layer and help facilitate predictions
    layers.Dense(64,activation='relu'),
    #layers.Dense(len(breeds)),
    layers.Dense(len(breeds), activation='softmax')
])

    # Compile the model
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=[
        'accuracy'])  # optimizer tells model how to predict error and how to iterate, and loss function calculates
    # error

    # fit the model
    history = model.fit(
        train,
        validation_data=test,
        epochs=10,
        verbose=1
    )
    data = {
        "train": train,
        "test": test,
    }
    flat_data = list(data.values())
    packed_data = tf.nest.pack_sequence_as(tf.nest.flatten(data), flat_data)
    unpacked_data = tf.nest.map_structure(lambda x: list(x.as_numpy_iterator()), packed_data)
    variable_retourne = (breeds, unpacked_data)
    f1 = open('node1tonode2picklefile', 'wb')
    pickle.dump(variable_retourne, f1)
    f1.close()
    history_df = pd.DataFrame.from_dict(history.history)
    return history_df
