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
    experiment_res_name = "experiment_convolutions_results_LONG"
    arr_models1 = {
        model18.make_model: "model 18 - 8 layers (32x8) 30 h lstm"
    }
    # модели участвующие в эксперименте
    arr_models = {
        model1.make_model: "model 1 - 1 layer (32x8) 50 h lstm",
        #model2.make_model: "model 2 - 1 layer (32x8) 30 h lstm",
        model3.make_model: "model 3 - 2 layers (32x8) 30 h lstm",
        model4.make_model: "model 4 - 3 layers (32x8) 30 h lstm",
        #model5.make_model: "model 5- 4 layers (16X5orx3) 30 h lstm",
        model6.make_model: "model 6 - 4 layers (32x8) 30 h lstm",
        model7.make_model: "model 7 - 4 layers (64x8) 30 h lstm---2",
        model8.make_model: "model 8 - 5 layers (32x8) 30 h lstm",
        model9.make_model: "model 9 - 6 layers (32x8) 30 h lstm",
        model10.make_model: "model 10- 7 layers (32x8) 30 h lstm",
        #model11.make_model: "model 11 - 7 layers (16x5) 15 h lstm",
        #model12.make_model: "model 12 - 7 layers 8x8 30 h lstm",
        #model13.make_model: "model 13 - 1 layer (32x8) 80 h lstm",
        model14.make_model: "model 14 - 1 layer (32x8) 60 h lstm",
        model15.make_model: "model 15 - 8 layers (8x8) 30 h lstm",
        model16.make_model: "model 16 - 7 layers (32X8) 30 h lstm---2",
        model17.make_model: "model 17 - 9 layers (32x8) 30 h lstm",
        model18.make_model: "model 18 - 8 layers (32x8) 30 h lstm"

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
    batch_size=25
    epochs=30


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
        stats = statistics(ytest[:, 1000:4000], pred_test[:, 1000:4000]).round(4)
        stats_dict[model_description] = stats
        stats.to_csv("stats_"+model_description + '.txt')
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