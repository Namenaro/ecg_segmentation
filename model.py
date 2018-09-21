import numpy as np
from keras import backend as K
from keras.callbacks import (
    TensorBoard
)
from keras.layers import (
    LSTM,
    Bidirectional,
    Dense,
    Conv1D,
    MaxPooling1D,
    UpSampling1D
)
from keras.models import (
    Sequential, load_model
)

def make_model(num_leads_signal):
    model = Sequential()

    model.add(Conv1D(32, kernel_size=8,
                     activation=K.elu,
                     input_shape=(None, num_leads_signal), padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(32, kernel_size=8, activation=K.elu, padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(32, kernel_size=8, activation=K.elu, padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(32, kernel_size=8, activation=K.elu, padding='same'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Bidirectional(LSTM(50, return_sequences=True)))

    model.add(UpSampling1D(2))
    model.add(Conv1D(32, kernel_size=8, activation=K.elu, padding='same'))
    model.add(UpSampling1D(2))
    model.add(Conv1D(32, kernel_size=8, activation=K.elu, padding='same'))
    model.add(UpSampling1D(2))
    model.add(Conv1D(32, kernel_size=8, activation=K.elu, padding='same'))
    model.add(UpSampling1D(2))
    model.add(Conv1D(32, kernel_size=8, activation=K.elu, padding='same'))
    model.add(Dense(4, activation='softmax'))

    return model