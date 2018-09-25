import os
import time
import shutil
import logging
import pandas as pd
from train import train
from experiment_freq.models import model_a
from sklearn.model_selection import train_test_split
import numpy as np

from crossvalidation import make_crossvalidation
from dataset import load_dataset


folder_for_results = "experiment_freq_results"

logging.basicConfig(filename='log.log', level=logging.DEBUG)

# создает отлельную папку под результаты эксперимента и делаем ее на время умолчательной
cwd = os.getcwd()
if os.path.exists(folder_for_results) and os.path.isdir(folder_for_results):
    shutil.rmtree(folder_for_results)

os.makedirs(folder_for_results)
os.chdir(folder_for_results)

# параметры эксперимента
win_len = 3072
batch_size=10
epochs=10
xy = load_dataset()
X = xy["x"]
Y = xy["y"]
X_freq_250 = np.zeros((X.shape[0],X.shape[1]//2,X.shape[2]))
Y_freq_250 = np.zeros((Y.shape[0],Y.shape[1]//2,Y.shape[2]))

X_freq_125 = np.zeros((X.shape[0],X.shape[1]//4,X.shape[2]))
Y_freq_125 = np.zeros((Y.shape[0],Y.shape[1]//4,Y.shape[2]))


for i in range(X.shape[0]):
    for k in range(X.shape[1]//2):
        for j in range(X.shape[2]):
            X_freq_250[i, k, j]= X[i, k*2, j]
        for j in range(Y.shape[2]):
            Y_freq_250[i, k, j]= Y[i, k*2, j]

    for k in range(X.shape[1]//4):
        for j in range(X.shape[2]):
            X_freq_125[i, k, j]= X[i, k*4, j]
        for j in range(Y.shape[2]):
            Y_freq_125[i, k, j]= Y[i, k*4, j]


arr_x = [
    X,
    X_freq_250,
    X_freq_125
]
arr_results = []
data_name = 0
for data in arr_x:
    logging.info("start freq= " + str(data_name) + " at " + str(time.ctime()))
    result= make_crossvalidation(kfold_splits=4,
                                 create_model=model_a.make_model,
                                 X=data, Y=Y,
                                 win_len=int(win_len//(2^data_name)),
                                 model_name='model'+str(data_name),
                                 batch_size=batch_size,
                                 epochs=epochs)
    logging.info(str(result))
    arr_results.append(result)
    data_name += 1

# сохраняем результаты в файл
table_results = pd.DataFrame(arr_results)
table_results.to_csv('freq_experiment_results.txt', header=True, index=True, sep='\t', mode='a')

print(table_results)
os.chdir(cwd)
