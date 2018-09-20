import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np

def save_history(history, name):
    name = "./pics/"+name
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(name+"_loss.png")
    plt.clf()

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

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, sharex=True)
    x = range(0, len(ecg_signal))

    ax1.fill_between(x, 0, prediction[:,0], where=prediction[:,0]>0.5, label="мнение сети",facecolor='green',alpha=0.5 )
    ax2.fill_between(x, 0, prediction[:,1], where=prediction[:,1]>0.5,label="мнение сети.",facecolor='green',alpha=0.5)
    ax3.fill_between(x, 0, prediction[:,2], where=prediction[:,2]>0.5, label="мнение сети.",facecolor='green',alpha=0.5)

    ax2.plot(prediction[:,0], 'k-', label="сырой отв.")
    ax2.plot(prediction[:,1], 'r-', label="сырой отв.")
    ax2.plot(prediction[:,2], 'g-', label="сырой отв.")


    ax1.fill_between(x,0,right_answer[:,0], alpha=0.5, label="правильн.отв.", facecolor='red')
    ax2.fill_between(x,0,right_answer[:,1], alpha=0.5, label="правильн.отв.", facecolor='red')
    ax3.fill_between(x,0,right_answer[:,2], alpha=0.5, label="правильн.отв.", facecolor='red')

    plt.legend(loc=2)
    plt.savefig(figname)
