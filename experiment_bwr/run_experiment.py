import os
import time
import shutil
import logging
import pandas as pd
from train import train
from experiment_bwr.models import model_a
import BaselineWanderRemoval as bwr
from sklearn.model_selection import train_test_split
import numpy as np

from crossvalidation import make_crossvalidation
from dataset import load_dataset

folder_for_results = "experiment_bwr_results"

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
epochs=15
xy = load_dataset()
X = xy["x"]
Y = xy["y"]
X_bwr = np.copy(X)

for i in range(X.shape[0]):
    for j in range(X.shape[2]):
        X_bwr[i, :, j] = bwr.fix_baseline_wander(X[i, :, j], 500)
xes = [X_bwr, X]
arr_results = []
mmm = 0
for data in xes:
    mmm += 1
    logging.info("start bwr= " + str(mmm) + " at " + str(time.ctime()))
    result= make_crossvalidation(kfold_splits=4,
                                 create_model=model_a.make_model,
                                 X=data, Y=Y,
                                 win_len=win_len,
                                 model_name='model'+str(mmm),
                                 batch_size=batch_size,
                                 epochs=epochs)
    logging.info(str(result))
    arr_results.append(result)

# сохраняем результаты в файл
table_results = pd.DataFrame(arr_results)
table_results.to_csv('bwr_experiment_results.txt', header=True, index=True, sep='\t', mode='a')

print(table_results)
os.chdir(cwd)