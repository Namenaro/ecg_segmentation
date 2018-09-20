import os
import json
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import pyedflib
import numpy as np
import pickle as pkl

# Порядок отведений
leads_names = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']

def load_dataset(description_data="C:\\ecg_new\\ecg_data_200.json"):
    with open(description_data, 'r') as f:
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

if __name__ == "__main__":
    xy = load_dataset()

