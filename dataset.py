import os
import json
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import pyedflib
import numpy as np
import pickle as pkl
import BaselineWanderRemoval as bwr

# Порядок отведений
leads_names = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
pkl_filename = "C:\\ecg_new\\dataset_fixed_baseline.pkl"
FREQUENCY_OF_DATASET = 500
raw_dataset_path="C:\\ecg_new\\ecg_data_200.json"

def load_raw_dataset(raw_dataset):
    with open(raw_dataset, 'r') as f:
        data = json.load(f)
    X=[]
    Y=[]
    for case_id in data.keys():
        leads = data[case_id]['Leads']
        x = []
        y = []
        for i in range(len(leads_names)):
            lead_name = leads_names[i]
            x.append(leads[lead_name]['Signal'])

        signal_len = 5000
        delineation_tables = leads[leads_names[0]]['DelineationDoc']
        p_delin = delineation_tables['p']
        qrs_delin = delineation_tables['qrs']
        t_delin = delineation_tables['t']

        p = get_mask(p_delin, signal_len)
        qrs = get_mask(qrs_delin, signal_len)
        t = get_mask(t_delin, signal_len)
        background = get_background(p, qrs, t)

        y.append(p)
        y.append(qrs)
        y.append(t)
        y.append(background)

        X.append(x)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    #X.shape = (200, 12, 5000)
    X = np.swapaxes(X, 1, 2)
    Y = np.swapaxes(Y, 1, 2)
    # X.shape = (200, 5000, 12)

    return {"x":X, "y":Y}

def get_mask(table, length):
    mask = [0] * length  # забиваем сначала маску нулями
    for triplet in table:
        start = triplet[0]
        end = triplet[2]
        for i in range(start, end, 1):
            mask[i] = 1
    return mask

def get_background(p, qrs, t):
    background = np.zeros_like(p)
    for i in range(len(p)):
        if p[i]==0 and qrs[i]==0 and t[i]==0:
            background[i]=1
    return background

def fix_baseline_and_save_to_pkl(xy):
    print("start fixing baseline in the whole dataset. It may take some time, wait...")
    X= xy["x"]
    for i in range(X.shape[0]):
        print(str(i))

        for j in range(X.shape[2]):
            X[i, :, j] = bwr.fix_baseline_wander(X[i, :, j], FREQUENCY_OF_DATASET)
    xy['x']=X
    outfile = open(pkl_filename, 'wb')
    pkl.dump(xy, outfile)
    outfile.close()
    print("dataset saved, number of pacients = " + str(len(xy['x'])))


def load_dataset(raw_dataset=raw_dataset_path, fixed_baseline=True):
    """
    при первом вызове с параметром fixed_baseline=True может работать очень долго, т.к. выполняет предобработку -
    затем резуотат предобрабоки сохраняется, чтоб не делать эту трудоемкую операцию много раз
    :param raw_dataset:
    :param fixed_baseline: флаг, нужно ли с выровненным дрейфом изолинии
    :return:
    """
    if fixed_baseline is True:
        print("you selected FIXED BASELINE WANDERING")
        if os.path.exists(pkl_filename): # если файл с предобработанным датасетом уже есть, не выполняем предобработку
            infile = open(pkl_filename, 'rb')
            dataset_with_fixed_baseline = pkl.load(infile)
            infile.close()
            return dataset_with_fixed_baseline
        else:
            xy = load_raw_dataset(raw_dataset) # если файл с обработанным датасетом еще не создан, создаем
            fix_baseline_and_save_to_pkl(xy)
            infile = open(pkl_filename, 'rb')
            dataset_with_fixed_baseline = pkl.load(infile)
            infile.close()
            return dataset_with_fixed_baseline
    else:
        print("you selected NOT fixied BASELINE WANDERING")
        return load_raw_dataset(raw_dataset)

if __name__ == "__main__":
    xy = load_dataset()
    print(xy)

