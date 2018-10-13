from scipy.stats import mannwhitneyu
import pickle as pkl
import pandas as pd
import numpy as np
import itertools
import os


pkl_filename = "C:\\data_csv\\stats_model.pkl"

points = ['p_start', 'p_end', 'qrs_start', 'qrs_end','p_start', 'p_end']
metrics = ['Se', 'PPV', 'm', 'sigma2']
number_of_leads = [1, 2, 12]

leads = dict(zip(list(range(len(number_of_leads))), number_of_leads))
number_of_experiments = 20

def data_of_experiments():
    sigma2_total = []
    PPV_total = []
    Se_total = []
    m_total = []
    for lead in leads.values():
        sigma2 = []
        PPV = []
        Se = []
        m = []
        for experiment in range(number_of_experiments):
            experiment_name = str(lead) + str(experiment)
            csv_file_path = "C:\\data_csv\\stats_model " + experiment_name + ".csv"

            data = pd.read_csv(csv_file_path)
            sigma2.append(data.iloc[3, 1:])
            PPV.append(data.iloc[1, 1:])
            Se.append(data.iloc[0, 1:])
            m.append(data.iloc[2, 1:])

        sigma2_total.append(sigma2)
        PPV_total.append(PPV)
        Se_total.append(Se)
        m_total.append(m)

    sigma2_total = np.array(sigma2_total)
    PPV_total = np.array(PPV_total)
    Se_total = np.array(Se_total)
    m_total = np.array(m_total)

    #metrica[leads][experiment][parametr]
    data = [Se_total, PPV_total, m_total, sigma2_total]
    dictionary = dict(zip(metrics, data))
    save_data_experiments_to_pkl(dictionary)

def save_data_experiments_to_pkl(dictionary):
    outfile = open(pkl_filename, 'wb')
    pkl.dump(dictionary, outfile)
    outfile.close()

def statistical_significance_of_differences(metrics, points, leads):
    if not os.path.exists(pkl_filename):
        data_of_experiments()
    infile = open(pkl_filename, 'rb')
    dictionary = pkl.load(infile)
    infile.close()

    alpha = 0.05

    for metric in metrics:
        Header = []
        Row = []
        Header.append(metric)
        Row.append(points)
        for (i, j) in list(itertools.combinations(leads.keys(), 2)):
            Header.append((leads[i], leads[j]))
            row = []
            for point in range(len(points)):
                stat, p = mannwhitneyu(dictionary[metric][i, :, point], dictionary[metric][j, :, point])
                if p > alpha: decision = 'H_0'
                else: decision = 'H_1'
                row.append(decision)

            Row.append(row)

        diction = dict(zip(Header, Row))
        print("\n")
        print(pd.DataFrame(diction))

    print('\nNumber of experiments: ' + str(number_of_experiments))
    print("H_0 : fail to reject null hypothesis (p > alpha) -> Sample distributions are equal.")
    print("H_1 : reject null hypothesis (p <= alpha) -> Sample distributions are not equal.")

statistical_significance_of_differences(metrics=metrics, points=points, leads=leads)