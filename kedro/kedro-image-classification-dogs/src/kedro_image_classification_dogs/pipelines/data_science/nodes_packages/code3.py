from .code2 import train_model
import tensorflow as tf
from tensorflow.keras import layers
import pickle


def troisieme_modele():
    f1 = open('node2tonode3picklefile', 'rb')
    variable_retourne = pickle.load(f1)
    f1.close()
    network = variable_retourne
    data_augmentaion = tf.keras.Sequential([
        layers.RandomFlip('horizontal', seed=1),
        # Randomly flips images from left to right which the model wuill see as a new image and increase accuracy
        layers.RandomRotation(.2, seed=1),  # Randomly raotates images for more information
        layers.RandomZoom(.2, seed=1)  # randomly zooms images for more information for the model
    ])
    full_network = [data_augmentaion] + network
    history_df, model = train_model(full_network)
    return history_df
