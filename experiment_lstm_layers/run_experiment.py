import os
import time
import shutil
import logging
import numpy as np
import pandas as pd
from train import train
from experiment_lstm_layers.models import *
from sklearn.model_selection import train_test_split
from crossvalidation import make_crossvalidation
from dataset import load_dataset
from metrics import *

folder_for_results = "experiment_lstm_layers_results"

logging.basicConfig(filename='log.log',level=logging.DEBUG)

# модели участвующие в эксперименте
arr_models = {
    modela.make_model:"model 1 lstm",
    modelb.make_model:"model 2 sltm",
    modelc.make_model:"model 3 lstm",
    modele.make_model:"model 4 lstm",
    modelf.make_model:"model 5 lstm"
}

# создает отлельную папку под результаты эксперимента и делаем ее на время умолчательной
cwd = os.getcwd()
if os.path.exists(folder_for_results) and os.path.isdir(folder_for_results):
    shutil.rmtree(folder_for_results)

os.makedirs(folder_for_results)
os.chdir(folder_for_results)

# параметры эксперимента
win_len = 3072
batch_size=10
epochs=14
xy = load_dataset()
X = xy["x"]
Y = xy["y"]

arr_results = []
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.33, random_state=42)
for make_model, model_description in arr_models.items():
    logging.info("start " + model_description + " at " + str(time.ctime()))
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

    result = {"avg_loss": history.history['loss'][-1],
              "avg_val_loss": history.history['val_loss'][-1],
              "avg_PPV": history.history['PPV'][-1],
              "avg_val_PPV": history.history['val_PPV'][-1],
              "avg_se": history.history['Se'][-1],
              "avg_val_se": history.history['val_Se'][-1]}
    logging.info(str(result))
    arr_results.append(result)
    pred_test = np.array(model.predict(xtest))
    stats = statistics(ytest[:, 1000:4000], pred_test[:, 1000:4000]).round(2)
    stats.to_csv("stats_" + model_description + '.txt')


# сохраняем результаты в файл
table_results = pd.DataFrame(arr_results)
table_results.to_csv('results.txt', header=True, index=True, sep='\t', mode='a')

print(table_results)
os.chdir(cwd)