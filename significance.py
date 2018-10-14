from scipy.stats import mannwhitneyu
import pickle as pkl
import pandas as pd
import numpy as np
import itertools
import os


# #LEADS
experiment_res_name = 'experiment_leads_'
pkl_filename = "C:\\data_csv\\stats_model.pkl"
number_of_leads = [1, 2, 12]

# #BASE WANDERING
# experiment_res_name = 'experiment_baseline_wandering_'
# pkl_filename = "C:\\data_csv\\stats_model_bw.pkl"
# number_of_leads = ['12', '12bw']

# #LSTM
# experiment_res_name = 'experiment_lstm_'
# pkl_filename = "C:\\data_csv\\stats_model_lstm.pkl"
# number_of_leads = ['0lstm', '1lstm', '2lstm', '3lstm'] #кол-во lstm (не отведения)

alpha = 0.05
number_of_experiments = 20
points = ['p_start', 'p_end', 'qrs_start', 'qrs_end','t_start', 't_end']
metrics = ['Se', 'PPV', 'm', 'sigma2']

leads = dict(zip(list(range(len(number_of_leads))), number_of_leads))
data_folder = "C:\\data_csv\\"

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
            csv_file_path = data_folder + "stats_model " + experiment_name + ".csv"

            data = pd.read_csv(csv_file_path, sep=';', encoding='latin1')
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

def statistical_significance_of_differences(metrics, points, leads, answer='hypothesis', alternative='two-sided'):
    """
        :param metrics: список метрик
        :param points: список рассматриваемых точек (начало/конец интервала)
        :param leads: список отведений или список кол-в lstm в сети
        :param answer: что вывести в таблицу: 'p' , 'hypothesis'
        :param alternative: альтернативная гипотеза для теста Манна-Уитни: 'less', 'greater', 'two-sided'
    """

    if not os.path.exists(pkl_filename):
        data_of_experiments()
    infile = open(pkl_filename, 'rb')
    dictionary = pkl.load(infile)
    infile.close()


    experiment_res_name_path = data_folder + experiment_res_name + str(alpha) + '_' + alternative + '_' + answer + '.csv'

    for metric in metrics:
        Header = []
        Row = []
        Header.append(metric)
        Row.append(points)
        for (i, j) in list(itertools.combinations(leads.keys(), 2)):
            Header.append((leads[i], leads[j]))
            row = []
            for point in range(len(points)):
                stat, p = mannwhitneyu(dictionary[metric][i, :, point], dictionary[metric][j, :, point], alternative=alternative)

                if p > alpha: decision = 'H_0'
                else: decision = 'H_1'

                if answer == 'p': row.append(round(p, 5))
                if answer == 'hypothesis': row.append(decision)

            Row.append(row)

        diction = dict(zip(Header, Row))
        table = pd.DataFrame(diction)
        table.loc[len(table)] = [''] * len(Header) #добавляется пустая строка, чтобы разделить метрики

        table.to_csv(experiment_res_name_path, sep=';', mode='a', index=False)

    print('\nThe result has been saved in ' + experiment_res_name_path + '.')
    print('\nNumber of experiments: ' + str(number_of_experiments))
    if answer == 'hypothesis':
        print("H_0 : fail to reject null hypothesis (p > alpha) -> Sample distributions are equal.")
        print("H_1 : reject null hypothesis (p <= alpha) -> Sample distributions are not equal.")

def get_statistics():
    if not os.path.exists(pkl_filename):
        data_of_experiments()
    infile = open(pkl_filename, 'rb')
    dictionary = pkl.load(infile)
    infile.close()

    for metric in metrics:
        Header = []
        Row = []
        Header.append(metric)
        Row.append(points)
        for i in leads.keys():
            Header.append(leads[i])
            row = []
            for point in range(len(points)):
                row.append((np.mean(dictionary[metric][i, :, point]).round(5), np.var(dictionary[metric][i, :, point]).round(5)))

            Row.append(row)

        diction = dict(zip(Header, Row))
        table = pd.DataFrame(diction)
        table.loc[len(table)] = [''] * len(Header)

        statistics_res_name_path = data_folder + '\\statistics.csv'
        table.to_csv(statistics_res_name_path, sep=';', mode='a', index=False)

    print('\nThe statistics has been saved in ' + data_folder + '\\statistics.csv' + '.')
    print('\nNumber of experiments: ' + str(number_of_experiments))

if __name__ == "__main__":
    statistical_significance_of_differences(metrics=metrics, points=points, leads=leads)
    # get_statistics()