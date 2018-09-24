import os
import time
import shutil
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from experiment_convolutions.models import *
from crossvalidation import make_crossvalidation
from dataset import load_dataset
from train import train
from metrics import statistics

def RUN():
    experiment_res_name = "experiment_convolutions_results"

    # модели участвующие в эксперименте
    arr_models = {
        model_a.make_model: "model a",
        model_b.make_model: "model b"

    }

    logging.basicConfig(filename='log.log',level=logging.DEBUG)
    logging.info(experiment_res_name)

    xy = load_dataset()
    X = xy["x"]
    Y = xy["y"]

    # создает отлельную папку под результаты эксперимента и делаем ее на время умолчательной
    cwd = os.getcwd()
    if os.path.exists(experiment_res_name) and os.path.isdir(experiment_res_name):
        shutil.rmtree(experiment_res_name)

    os.makedirs(experiment_res_name)
    os.chdir(experiment_res_name)

    # common parameters in all models:
    win_len = 3072
    batch_size=3
    epochs=1


    arr_summaries = []
    stats_dict = {}
    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.33, random_state=42)

    # iterate through all the models...
    for make_model, model_description in arr_models.items():
        try:
            model = None
            model = make_model()

            history = train(model,
                            model_name=model_description,
                            x_test=xtest,
                            x_train=xtrain,
                            y_test=ytest,
                            y_train=ytrain,
                            win_len=win_len,
                            batch_size=batch_size,
                            epochs=epochs)

            summary = {"model_name": model_description,
                      "loss": history.history['loss'][-1],
                      "val_loss": history.history['val_loss'][-1],
                      "PPV": history.history['PPV'][-1],
                      "val_PPV": history.history['val_PPV'][-1],
                      "se": history.history['Se'][-1],
                      "val_se": history.history['val_Se'][-1]}
        except Exception:
            logging.error("ERROR OCCURED IN MODEL " + model_description)
            continue

        logging.info(str(summary))
        arr_summaries.append(summary)

        pred_test = np.array(model.predict(xtest))
        stats = statistics(ytest[:, 1000:4000], pred_test[:, 1000:4000]).round(2)
        stats_dict[model_description] = stats
        stats.to_csv("stats_"+model_description + '.csv')
        print(stats)

    # save results into file:
    table_summaries = pd.DataFrame(arr_summaries)
    table_summaries.to_csv(experiment_res_name+'.txt', header=True, index=True, sep='\t', mode='a')
    logging.info("STATISTICS------------------------------------------------------")
    logging.info(stats_dict)
    print(table_summaries)
    os.chdir(cwd)


if __name__ == "__main__":
    RUN()