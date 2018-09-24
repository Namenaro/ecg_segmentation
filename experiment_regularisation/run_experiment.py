import os
import time
import shutil
import logging
import pandas as pd

from experiment_regularisation.models import *

from crossvalidation import make_crossvalidation
from dataset import load_dataset

folder_for_results = "experiment_regularisation_results"

logging.basicConfig(filename='log.log',level=logging.DEBUG)

# модели участвующие в эксперименте
arr_models = {
    modela.make_model:"model a",
    modela_reg.make_model: "model a reg",
    modelb.make_model:"model b",
    modelb_reg.make_model:"model b reg",
    modelc.make_model:"model c",
    modelc_reg.make_model:"modelc_reg",
    modelc_reg_drop.make_model:"modelc_reg_drop",
    modelc_rec_drop2.make_model:"modelc_rec_drop2",
    modelc_rec_drop3.make_model:"modelc_rec_drop3",
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
epochs=13
xy = load_dataset()
X = xy["x"]
Y = xy["y"]

arr_results = []
# эксперимент - кроссвалидация по всем моделям
for make_model, model_description in arr_models.items():
    logging.info("start crossvalidation " + model_description + " at " + str(time.ctime()))
    result= make_crossvalidation(kfold_splits=4,
                         create_model=make_model,
                         X=X, Y=Y,
                         win_len=win_len,
                         model_name=model_description,
                         batch_size=batch_size,
                         epochs=epochs)
    logging.info(str(result))
    arr_results.append(result)

# сохраняем результаты в файл
table_results = pd.DataFrame(arr_results)
table_results.to_csv('regularisation_experiment_results.txt', header=True, index=True, sep='\t', mode='a')

print(table_results)
os.chdir(cwd)