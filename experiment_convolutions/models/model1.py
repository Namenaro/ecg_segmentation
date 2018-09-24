from keras import backend as K
from keras.layers import (
    LSTM,
    Bidirectional,
    Dense,
    Conv1D,
    MaxPooling1D,
    UpSampling1D
)
from keras.models import (
    Sequential
)

from metrics import Metrics

def make_model():
    num_leads_signal = 12
    model = Sequential()




    model.add(Bidirectional(LSTM(50, return_sequences=True)))



    model.add(Dense(4, activation='softmax'))

    metric = Metrics()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy', metric.Se, metric.PPV])
    return model