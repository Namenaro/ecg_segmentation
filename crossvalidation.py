from sklearn.model_selection import KFold
import numpy as np

from train import train
from experiment_convolutions.models import model_c
from dataset import load_dataset


def make_crossvalidation(kfold_splits, create_model, X, Y, win_len, model_name, batch_size, epochs):
    """

    :param kfold_splits: сколько фолдов в кроссвалидации (на сколько частей дели весь датасет)
    :param create_model: функция, возвращающая скомпиллированную модель
    :param X: весь датасет - экгшки
    :param Y: весь датасет -разметки
    :return:
    """
    kf = KFold(n_splits=kfold_splits, shuffle=True)
    indices = kf.split(X=X,y=Y)

    arr_Se = []
    arr_val_Se = []
    arr_loss = []
    arr_val_loss = []
    arr_PPV = []
    arr_val_PPV = []

    for index, (train_indices, val_indices) in enumerate(indices):
        xtrain, xtest = X[train_indices], X[val_indices]
        ytrain, ytest = Y[train_indices], Y[val_indices]

        print("Training on fold " + str(index + 1) + " from " + str(kfold_splits))
        print(str(xtrain.shape[0]) + " training samples, " + str(xtest.shape[0]) + " validation samples")

        model = create_model()

        history = train(model,
              model_name=model_name+str(index+1),
              x_test=xtest,
              x_train=xtrain,
              y_test=ytest,
              y_train=ytrain,
              win_len=win_len,
              batch_size=batch_size,
              epochs=epochs)
        arr_loss.append(history.history['loss'][-1])
        arr_val_loss.append(history.history['val_loss'][-1])
        arr_Se.append(history.history['Se'][-1])
        arr_val_Se.append(history.history['val_Se'][-1])
        arr_PPV.append(history.history['PPV'][-1])
        arr_val_PPV.append(history.history['val_PPV'][-1])
    result = {"avg_loss": np.array(arr_loss).mean(),
              "avg_val_loss" : np.array(arr_val_loss).mean(),
              "avg_PPV": np.array(arr_PPV).mean(),
              "avg_val_PPV": np.array(arr_val_PPV).mean(),
              "avg_se": np.array(arr_Se).mean(),
              "avg_val_se" : np.array(arr_val_Se).mean()}
    return result

if __name__ == "__main__":
    # пример вызова
    win_len = 2992
    name = "mymodel_c"
    batch_size=10
    epochs=2

    xy = load_dataset()
    X = xy["x"]
    Y = xy["y"]

    result= make_crossvalidation(kfold_splits=3,
                         create_model=model_c.make_model,
                         X=X, Y=Y,
                         win_len=win_len,
                         model_name=name,
                         batch_size=batch_size,
                         epochs=epochs)
    print(result)
