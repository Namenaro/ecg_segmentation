# найдем все модели в выбранной папке и померим для них статистику, после чего распечатаем имена моделей в порядке убывания качества

import easygui
import os
from utils import restore_set_from_pkl
from keras.models import load_model
from metrics import *

def get_stats_from_folder_with_models():
    path_to_folder = easygui.diropenbox("выберите папку")
    models_and_their_stats = {}
    for file in os.listdir(path_to_folder):
        filename = os.fsdecode(file)
        if filename.endswith(".h5"):
            case_id = filename[0:-len(".h5")]
            pkl_filename = case_id + ".pkl"
            pkl_filename = os.path.join(path_to_folder, pkl_filename)
            print(pkl_filename)
            X_test, Y_test = restore_set_from_pkl(pkl_filename)
            metric = Metrics()
            Se = metric.Se
            PPV = metric.PPV
            filename = os.path.join(path_to_folder, filename)
            model = load_model(filename, custom_objects={'Se': Se, 'PPV': PPV})
            pred_test = np.array(model.predict(X_test))

            frame = statistics(Y_test[:, 1000:4000], pred_test[:, 1000:4000]).round(4)
            print(frame)
            models_and_their_stats[case_id]= frame
            model=None
    return models_and_their_stats

def compare_models_by_stats(models_and_their_stats):
    pass

if __name__ == "__main__":
    models_and_their_stats = get_stats_from_folder_with_models()
    print(models_and_their_stats)

