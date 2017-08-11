import numpy as np
from scipy.spatial.distance import cdist

from keras.layers import Input, Dense, Dropout
from keras.models import Model

class Autoencoder(object):
    """docstring for Autoencoder."""
    def __init__(self):
        super(Autoencoder, self).__init__()


    def fit(self, data):
        _input = Input(shape=(data.shape[1],))
        encoded = Dense(128, activation='relu')(_input)
        encoded = Dropout(0.5)(encoded)
        encoded = Dense(64, activation='relu')(encoded)
        encoded = Dropout(0.3)(encoded)
        encoded = Dense(32, activation='relu')(encoded)

        decoded = Dense(64, activation='relu')(encoded)
        decoded = Dense(128, activation='relu')(decoded)
        decoded = Dense(data.shape[1], activation='sigmoid')(decoded)

        autoencoder = Model(_input, decoded)

        autoencoder.fit(x_train[:int(-data.shape[0]/3)], x_train[:int(-data.shape[0]/3)],
                        epochs=1000,
                        batch_size=256,
                        shuffle=True,
                        validation_data=(data[:int(data.shape[0]/3)], data[:int(data.shape[0]/3)]))

        self.model = autoencoder

    def predict(self, _in):
        return self.model.predict(_in)
