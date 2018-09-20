import numpy as np

def generator(X,Y, win_len, batch_size, num_leads_signal=12):
    """

    :param X: экгшки (все 12 отведений)
    :param Y: соотв. им докторская разметка
    :param win_len: какиой длины куски экг вырезать
    :param batch_size: сколько пациентов брать в батч
    :return:
    """

    all_ecg_len = X.shape[1]
    num_pacients = X.shape[0]

    while True:
        batch_x = []
        batch_y = []
        for i in range(0, batch_size):
            starting_position = np.random.randint(0, all_ecg_len - win_len)
            ending_position = starting_position + win_len
            rand_pacient_id = np.random.randint(0, num_pacients)

            x = X[rand_pacient_id, starting_position:ending_position, 0:num_leads_signal]
            y = Y[rand_pacient_id, starting_position:ending_position, :]
            batch_x.append(x)
            batch_y.append(y)

        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        yield (batch_x, batch_y)