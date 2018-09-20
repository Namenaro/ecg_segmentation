import matplotlib.pyplot as plt

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
