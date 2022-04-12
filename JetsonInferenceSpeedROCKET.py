import gc
import multiprocessing
import warnings

import numpy as np
from numba import set_num_threads

warnings.filterwarnings("ignore")

import glob
from timeit import default_timer as timer

from sktime.datatypes._panel._convert import from_3d_numpy_to_nested
from sktime.transformations.panel.rocket import Rocket


def measure_transform_times():
    """
        Loads the datasets, trains a ROCKET model with seed 0 on each dataset and measures the time required for transforming a random test sample. The measured times are saved to a csv file.
    """
    global_times = []
    for filename in sorted(glob.glob(F"UEA/*.npz")):
        local_times = []
        dataset = filename.split("/")[-1].split(".")[0]
        data = np.load(filename)
        train_x, test_x = data['train_x'].astype(np.float64), data['test_x'].astype(np.float64)
        train_x = from_3d_numpy_to_nested(train_x)

        rocket = Rocket(random_state=0, n_jobs=-1)
        set_num_threads(multiprocessing.cpu_count())
        rocket.fit(train_x)
        with open("sum_channels.txt", "a") as of:
            of.write(f"{dataset},{rocket.kernels[-2].sum()}\n")
        for i in range(100):
            print(f'{dataset} - Rep: {i}')
            ind = np.random.choice(test_x.shape[0])
            tr_test = from_3d_numpy_to_nested(test_x[ind:ind + 1])
            gc.collect()
            start = timer()
            X_test_transform = rocket.transform(tr_test)
            end = timer()
            local_times.append(end - start)
        global_times.append(local_times)
    np.savetxt(F"jetson_rocket_transform_times.csv", np.array(global_times), delimiter=',')


if __name__ == "__main__":
    measure_transform_times()
