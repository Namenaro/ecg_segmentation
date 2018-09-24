import os
import time
import shutil
import logging
import pandas as pd
from train import train
from experiment_noise.models import model_a
from sklearn.model_selection import train_test_split
import numpy as np

from crossvalidation import make_crossvalidation
from dataset import load_dataset
import pywt
from statsmodels.robust import mad

def waveletSmooth( x, wavelet="db4", level=1, title=None ):
    # calculate the wavelet coefficients
    coeff = pywt.wavedec( x, wavelet, mode="per" )
    # calculate a threshold
    sigma = mad( coeff[-level] )
    # changing this threshold also changes the behavior,
    # but I have not played with this very much
    uthresh = sigma * np.sqrt( 2*np.log( len( x ) ) )
    coeff[1:] = ( pywt.threshold( i, value=uthresh, mode="soft" ) for i in coeff[1:] )
    # reconstruct the signal using the thresholded coefficients
    y = pywt.waverec( coeff, wavelet, mode="per" )
    return y

folder_for_results = "experiment_noise_results"

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
X_noise = np.copy(X)

#for i in range(X.shape[0]):
#    for j in range(X.shape[2]):
#        X_noise[i, :, j]= waveletSmooth(X[i, :, j], wavelet="db4", level=1, title=None )
xes = [X_noise, X]
arr_results = []
mmm = 0
for data in xes:
    mmm += 1
    logging.info("start noise= " + str(mmm) + " at " + str(time.ctime()))
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
table_results.to_csv('noise_experiment_results.txt', header=True, index=True, sep='\t', mode='a')

print(table_results)
os.chdir(cwd)