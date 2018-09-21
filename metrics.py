import numpy as np
import tensorflow as tf

freq = 500  # частота дискретизации
tolerance = (150 / 1000) * freq  # допустимое временное окно 150 мс


def change_mask(sample):
    """
    приводим маску к виду, где хранится только начало и конец интервала
    :param sample:
    :return:
    """
    p = [[], []]
    qrs = [[], []]
    t = [[], []]
    for i in range(sample.shape[0] - 1):
        if sample[i] != 1 and sample[i + 1] == 1:
            # start_P
            p[0].append(i)
        elif sample[i] != 2 and sample[i + 1] == 2:
            # start_QRS
            qrs[0].append(i)
        elif sample[i] != 3 and sample[i + 1] == 3:
            # start_T
            t[0].append(i)

        if sample[i] == 1 and sample[i + 1] != 1:
            # end_P
            p[1].append(i)
        elif sample[i] == 2 and sample[i + 1] != 2:
            # end_QRS
            qrs[1].append(i)
        elif sample[i] == 3 and sample[i + 1] != 3:
            # end_T
            t[1].append(i)
    return p, qrs, t


def comprassion(mask1, mask2, start_or_end):
    """
    сравнивает два отведения по одному полю
    :param mask1:
    :param mask2:
    :param start_or_end: 0 -- начало интервала, 1 -- конец
    :return:
    """
    tp = []
    fp = []
    fn = []
    error = []

    for p_1 in mask1[start_or_end]:
        flag = False
        for p_2 in mask2[start_or_end]:
            if p_1 + tolerance >= p_2 > p_1 - tolerance:
                tp.append(p_1)
                error.append((p_1 - p_2) / freq)
                mask2[start_or_end].remove(p_2)
                flag = True
                break
        if not flag:
            fp.append(p_1)

    for p_2 in mask2[start_or_end]:
        fn.append(p_2)

    return tp, fp, fn, error


class Metrics(object):
    def Se(self, y_true, y_pred):
        return tf.py_func(self._np_Se, [y_true, y_pred], tf.float32)

    @staticmethod
    def _np_Se(y_true, y_pred):
        true_pos = 0
        false_neg = 0

        for j in range(y_pred.shape[0]):
            sample_true = np.argmax(y_true[j], 1)
            sample_pred = np.argmax(y_pred[j], 1)

            p1, qrs1, t1 = change_mask(sample_pred)
            p2, qrs2, t2 = change_mask(sample_true)

            for i in range(2):
                tp, _, fn, _ = comprassion(p1, p2, i)
                true_pos += len(tp)
                false_neg += len(fn)

                tp, _, fn, _ = comprassion(qrs1, qrs2, i)
                true_pos += len(tp)
                false_neg += len(fn)

                tp, _, fn, _ = comprassion(t1, t2, i)
                true_pos += len(tp)
                false_neg += len(fn)

                if true_pos + false_neg == 0:
                    res = 0
                else:
                    res = true_pos / (true_pos + false_neg)

        return np.mean(res).astype(np.float32)

    def PPV(self, y_true, y_pred):
        return tf.py_func(self._np_PPV, [y_true, y_pred], tf.float32)

    @staticmethod
    def _np_PPV(y_true, y_pred):
        true_pos = 0
        false_pos = 0

        for j in range(y_pred.shape[0]):
            sample_true = np.argmax(y_true[j], 1)
            sample_pred = np.argmax(y_pred[j], 1)

            p1, qrs1, t1 = change_mask(sample_pred)
            p2, qrs2, t2 = change_mask(sample_true)

            for i in range(2):
                tp, fp, _, _ = comprassion(p1, p2, i)
                true_pos += tp
                false_pos += fp

                tp, fp, _, _ = comprassion(qrs1, qrs2, i)
                true_pos += tp
                false_pos += fp

                tp, fp, _, _ = comprassion(t1, t2, i)
                true_pos += tp
                false_pos += fp

                if true_pos + false_pos == 0:
                    res = 0
                else:
                    res = true_pos / (true_pos + false_pos)

        return np.mean(res).astype(np.float32)

    def mean(self, y_true, y_pred):
        return tf.py_func(self._np_mean, [y_true, y_pred], tf.float32)

    @staticmethod
    def _np_mean(y_true, y_pred):
        true_pos = 0
        false_pos = 0

        for j in range(y_pred.shape[0]):
            sample_true = np.argmax(y_true[j], 1)
            sample_pred = np.argmax(y_pred[j], 1)

            p1, qrs1, t1 = change_mask(sample_pred)
            p2, qrs2, t2 = change_mask(sample_true)

            for i in range(2):
                tp, fp, _, _ = comprassion(p1, p2, i)
                true_pos += tp
                false_pos += fp

                tp, fp, _, _ = comprassion(qrs1, qrs2, i)
                true_pos += tp
                false_pos += fp

                tp, fp, _, _ = comprassion(t1, t2, i)
                true_pos += tp
                false_pos += fp

                if true_pos + false_pos == 0:
                    res = 0
                else:
                    res = true_pos / (true_pos + false_pos)

        return np.mean(res).astype(np.float32)
