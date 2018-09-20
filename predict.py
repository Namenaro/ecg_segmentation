import matplotlib.pyplot as plt
import numpy as np
import easygui

from keras.models import load_model
from utils import *

def predict():
    filepath = easygui.fileopenbox("select model (.h5)")
    dataset_path = filepath[0:-2] + "pkl"

    mymodel = load_model(filepath)
    X, Y = restore_set_from_pkl(dataset_path)
    predict_in_one_pacient(X[0], Y[0], mymodel)

def predict_in_one_pacient(x, y, model):
    time_len = model.input_shape[1]
    num_leads = model.input_shape[2]

    batch_x, batch_y = split_ecg_in_patches(x, y, time_len)
    y_predicted = model.predict_on_batch(batch_x)
    draw_prediction_and_reality(batch_x[0], prediction=y_predicted[0],
                                right_answer=batch_y[0],
                                plot_name="predicted")

def split_ecg_in_patches(x, y, time_len):
    batch_x = []
    batch_y = []
    full_time_len = len(x)
    assert full_time_len >= time_len

    time_pointer = 0
    while time_pointer + time_len < full_time_len:
        patch_x = x[time_pointer:time_pointer + time_len]
        patch_y = y[time_pointer:time_pointer + time_len]
        batch_x.append(patch_x)
        batch_y.append(patch_y)
        time_pointer += time_len
    if time_pointer < full_time_len - 1:
        last_patch_x = x[full_time_len - 1 - time_pointer:full_time_len - 1]
        last_patch_y = y[full_time_len - 1 - time_pointer:full_time_len - 1]
        batch_x.append(last_patch_x)
        batch_y.append(last_patch_y)
    return np.array(batch_x), np.array(batch_y)


if __name__ == "__main__":
    predict()