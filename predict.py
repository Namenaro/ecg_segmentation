import matplotlib.pyplot as plt
import numpy as np
import easygui

from keras.models import load_model
from utils import *
from metrics import Metrics
def predict():
    filepath = easygui.fileopenbox("select model (.h5)")
    dataset_path = filepath[0:-2] + "pkl"

    metric = Metrics()
    Se = metric.Se
    PPV = metric.PPV
    model = load_model(filepath, custom_objects={'Se': Se, 'PPV':PPV})
    X, Y = restore_set_from_pkl(dataset_path)

    y_predicted = model.predict_on_batch(X)

    time_len = len(y_predicted[1,:])
    for i in range(len(X)):

        draw_prediction_and_reality(X[i,0:time_len,:],
                                    prediction=y_predicted[i],
                                    right_answer=Y[i,0:time_len,:],
                                    plot_name="predicted" + str(i))



if __name__ == "__main__":
    predict()