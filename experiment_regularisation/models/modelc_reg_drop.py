from keras import backend as K
from keras.layers import (
    LSTM,
    Bidirectional,
    Dense,
    Conv1D,
    MaxPooling1D,
    UpSampling1D,
    Dropout
)
from keras.models import (
    Sequential
)

from metrics import Metrics

droput_rate = 0.2
def make_model():
    num_leads_signal = 12
    model = Sequential()

    model.add(Conv1D(32, kernel_size=8,
                     activation=K.elu,
                     input_shape=(None, num_leads_signal), padding='same'))
    model.add(Dropout(rate=droput_rate))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(32, kernel_size=8, activation=K.elu, padding='same'))
    model.add(Dropout(rate=droput_rate))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(32, kernel_size=8, activation=K.elu, padding='same'))
    model.add(Dropout(rate=droput_rate))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(32, kernel_size=8, activation=K.elu, padding='same'))
    model.add(Dropout(rate=droput_rate))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Bidirectional(LSTM(20, return_sequences=True)))

    model.add(UpSampling1D(2))
    model.add(Conv1D(32, kernel_size=8, activation=K.elu, padding='same'))
    model.add(Dropout(rate=droput_rate))
    model.add(UpSampling1D(2))
    model.add(Conv1D(32, kernel_size=8, activation=K.elu, padding='same'))
    model.add(Dropout(rate=droput_rate))
    model.add(UpSampling1D(2))
    model.add(Conv1D(32, kernel_size=8, activation=K.elu, padding='same'))
    model.add(Dropout(rate=droput_rate))
    model.add(UpSampling1D(2))
    model.add(Conv1D(32, kernel_size=8, activation=K.elu, padding='same'))
    model.add(Dropout(rate=droput_rate))
    model.add(Dense(4, activation='softmax'))

    metric = Metrics()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy', metric.Se, metric.PPV])
    return model