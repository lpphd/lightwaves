import gc
import multiprocessing
import warnings

from numba import set_num_threads
import numpy as np

import glob
from timeit import default_timer as timer

warnings.filterwarnings("ignore")

from sktime.datatypes._panel._convert import from_3d_numpy_to_nested
from sktime.transformations.panel.rocket import MiniRocketMultivariate


def measure_transform_times():
    """
    Loads the datasets, trains a MINIROCKET model with seed 0 on each dataset and measures the time required for transforming a random test sample. The measured times are saved to a csv file.
    """
    global_times = []
    for filename in sorted(glob.glob(F"UEA/*.npz")):
        local_times = []
        data = np.load(filename)
        dataset = filename.split("/")[-1].split(".")[0]
        train_x, test_x = data['train_x'].astype(np.float64), data['test_x'].astype(np.float64)

        if train_x.shape[-1] < 9:
            train_x = np.pad(train_x, ((0, 0), (0, 0), (0, 9 - train_x.shape[-1])))
            test_x = np.pad(test_x, ((0, 0), (0, 0), (0, 9 - train_x.shape[-1])))

        train_x = from_3d_numpy_to_nested(train_x)

        minirocket = MiniRocketMultivariate(random_state=0, n_jobs=-1)
        set_num_threads(multiprocessing.cpu_count())
        minirocket.fit(train_x)
        for i in range(100):
            print(f'{dataset} - Rep: {i}')
            ind = np.random.choice(test_x.shape[0])
            tr_test = from_3d_numpy_to_nested(test_x[ind:ind + 1])
            gc.collect()
            start = timer()
            X_test_transform = minirocket.transform(tr_test)
            end = timer()
            local_times.append(end - start)
        global_times.append(local_times)
    np.savetxt(F"jetson_minirocket_transform_times.csv", np.array(global_times), delimiter=',')


if __name__ == "__main__":
    measure_transform_times()
