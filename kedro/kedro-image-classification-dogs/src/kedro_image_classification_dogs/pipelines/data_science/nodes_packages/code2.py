import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import tensorflow as tf
import pickle


# Account for overfitting and increase model accuracy
# paste model from before into function to not have to change the code every time

def train_model(network, test, train, epochs=5) -> pd.DataFrame:
    model = Sequential(network)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(
        train,
        validation_data=test,
        epochs=10,
        verbose=1
    )

    # Création du dataframe à partir du dictionnaire
    history_df = pd.DataFrame.from_dict(history.history)
    return history_df, model


def deuxieme_modele():
    f1 = open('node1tonode2picklefile', 'rb')
    variable_retourne = pickle.load(f1)
    f1.close()
    breeds, loaded_data = variable_retourne
    packed_loaded_data = tf.nest.map_structure(lambda x: tf.data.Dataset.from_tensor_slices(x), loaded_data)
    flattened_loaded_data = tf.nest.flatten(packed_loaded_data)
    train = flattened_loaded_data[0]
    test = flattened_loaded_data[1]
    print(type(test))
    network = [
    layers.Rescaling(1./255),
    layers.Conv2D(16,4,padding='same',activation='relu',input_shape=(224,224,3)),     # increase window size to 4 from 3
    layers.MaxPooling2D(),                                                            # add max pooling 2d layer to reduce overfit and reduce number of parameters
    layers.Conv2D(32,4,padding='same',activation='relu',input_shape=(224,224,3)),     # add second convolutional layer with increased filters to 32, to let network pick up higher level features
    layers.MaxPooling2D(),                                                           # add another max pooling layer
    layers.Conv2D(64,4,padding='same',activation='relu',input_shape=(224,224,3)),    # add another convolutional layer with 64 filters for even higher level features
    layers.MaxPooling2D(),                                                           # another max pooling layer
    layers.Dropout(.2),                                                              # Dropout layer helps with overfitting by setting some outputs to 0 randomly, so network doesnt become too linked to trainjing data
    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dense(64,activation='relu'),
    layers.Dense(len(breeds), activation='softmax')
]

    history_df, model = train_model(network, test, train)
    variable_retourne = network
    f1 = open('node2tonode3picklefile', 'wb')
    pickle.dump(variable_retourne, f1)
    f1.close()
    return history_df
