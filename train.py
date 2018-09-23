import os

from sklearn.model_selection import train_test_split

from dataset import load_dataset
from experiment_convolutions.models.model_c import make_model
from generator import generator
from utils import *


def train(model, model_name, x_test, x_train, y_test, y_train, win_len, batch_size, epochs):
    model.summary()
    num_leads_signal = model.input_shape[2]
    train_generator = generator(X=x_train, Y=y_train, win_len=win_len, batch_size=batch_size, num_leads_signal=num_leads_signal)
    test_set = next(generator(X=x_test, Y=y_test, win_len=win_len, batch_size=300, num_leads_signal=num_leads_signal))
    history = model.fit_generator(train_generator,
                                  epochs=epochs,
                                  steps_per_epoch=10,
                                  validation_data=(test_set[0], test_set[1]))

    folder_name = "trained_models"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    model.save(os.path.join(folder_name, model_name + '.h5'))
    save_set_to_pkl(x_test, y_test, os.path.join(folder_name, model_name + ".pkl"))
    save_history(history, model_name)
    return history

if __name__ == "__main__":
    # пример использования
    xy = load_dataset()
    X = xy["x"]
    Y = xy["y"]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    model = make_model()
    win_len = 3072
    name = "mymodel_c"
    train(model,
          model_name=name,
          x_test=X_test,
          x_train=X_train,
          y_test=Y_test,
          y_train=Y_train,
          win_len=win_len,
          batch_size=10,
          epochs=2)
