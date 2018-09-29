import os
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np

def save_history(history, name):
    folder_name = "pics"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['Se'])
    plt.plot(history.history['val_Se'])
    plt.plot(history.history['val_PPV'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train loss', 'test loss', 'train se','test se', 'val ppv'], loc='upper left')
    plt.savefig(os.path.join(folder_name, name+"_loss.png"))
    plt.clf()
    h = {}
    h['loss']=history.history['loss']
    h['val_loss'] = history.history['val_loss']
    h['se'] = history.history['Se']
    h['val_se'] = history.history['val_Se']
    outfile = open(name+'history.pkl', 'wb')
    pkl.dump(h, outfile)
    outfile.close()


def restore_set_from_pkl(path):
    infile = open(path, 'rb')
    load_pkl = pkl.load(infile)
    infile.close()
    return load_pkl['x'], load_pkl['y']

def save_set_to_pkl(x, y, name_pkl):
    assert len(x) == len(y)
    dict = {'x': np.array(x), 'y': np.array(y)}
    outfile = open(name_pkl, 'wb')
    pkl.dump(dict, outfile)
    outfile.close()
    print("dataset saved, number of pacients = " + str(len(x)))

def draw_prediction_and_reality(ecg_signal, prediction, right_answer, plot_name):
    """

    :param ecg_signal: сигнал некотего отведения
    :param prediction: предсказаные бинарные маски для этого отведения
    :param right_answer: правильная маска этого отведения (тоже три штуки)
    :param plot_name: имя картинки, куда хотим отрисовать это
    :return:
    """
    figname = plot_name + "_.png"
    print(ecg_signal.shape)
    print(prediction.shape)
    print(right_answer.shape)
    f, (ax1, ax2) = plt.subplots(2, 1, sharey=False, sharex=True, figsize=(20, 5))
    x = range(0, len(ecg_signal))

    ax1.plot(ecg_signal[:,0], color='black')

    ax1.fill_between(x, 0, 100, where=right_answer[:, 0]>0.5, alpha=0.5, color='red')
    ax1.fill_between(x, 0, 100, where=right_answer[:, 1]>0.5, alpha=0.5, color='green')
    ax1.fill_between(x, 0, 100, where=right_answer[:, 2]>0.5, alpha=0.5, color='blue')

    ax1.fill_between(x, 120, 220, where=prediction[:, 0] > 0.5, alpha=0.8, color='red')
    ax1.fill_between(x, 120, 220, where=prediction[:, 1] > 0.5, alpha=0.8, color='green')
    ax1.fill_between(x, 120, 220, where=prediction[:, 2] > 0.5, alpha=0.8, color='blue')

    ax2.plot(prediction[:,0], 'r-')
    ax2.plot(prediction[:,1], 'g-')
    ax2.plot(prediction[:,2], 'b-')

    plt.legend(loc=2)
    plt.savefig(figname)
    plt.clf()
