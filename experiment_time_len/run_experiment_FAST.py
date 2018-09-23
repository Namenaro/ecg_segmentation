import os
import time
import shutil
import logging
import pandas as pd
from train import train
from experiment_time_len.models import modela
from sklearn.model_selection import train_test_split
from crossvalidation import make_crossvalidation
from dataset import load_dataset

folder_for_results = "experiment_time_len"


logging.basicConfig(filename='log.log',level=logging.DEBUG)

time_lens = [256, 512, 1024, 2048, 4096]

# создает отлельную папку под результаты эксперимента и делаем ее на время умолчательной
cwd = os.getcwd()
if os.path.exists(folder_for_results) and os.path.isdir(folder_for_results):
    shutil.rmtree(folder_for_results)

os.makedirs(folder_for_results)
os.chdir(folder_for_results)

# параметры эксперимента

batch_size=10
epochs=13
xy = load_dataset()
X = xy["x"]
Y = xy["y"]

arr_results = []

for win_len in time_lens:
    logging.info("start win_len= " + str(win_len)+ " at " + str(time.ctime()))
    model = modela.make_model()
    model_name = "modela_"+ str(win_len)
    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.33, random_state=42)
    history = train(model,
                    model_name=model_name,
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

# сохраняем результаты в файл
table_results = pd.DataFrame(arr_results)
table_results.to_csv('results.txt', header=True, index=True, sep='\t', mode='a')

print(table_results)
os.chdir(cwd)