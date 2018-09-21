import matplotlib.pyplot as plt
import numpy as np
import easygui

from keras.models import load_model
from utils import *

def predict():
    filepath = easygui.fileopenbox("select model (.h5)")
    dataset_path = filepath[0:-2] + "pkl"

    model = load_model(filepath)
    X, Y = restore_set_from_pkl(dataset_path)
    num_leads = model.input_shape[2]
    y_predicted = model.predict_on_batch(X)

    time_len = len(y_predicted[0,:])
    draw_prediction_and_reality(X[0,0:time_len,:],
                                prediction=y_predicted[0],
                                right_answer=Y[0,0:time_len,:],
                                plot_name="predicted")



if __name__ == "__main__":
    predict()