import gc
import pickle

import numpy as np

import glob
from timeit import default_timer as timer
from numba import set_num_threads
import multiprocessing
from sympy.utilities.iterables import multiset_permutations
from lightwavesl2_functions import _apply_2layer_kernels


def GetScaleParams(test_x):
    """
    Returns mean and std per channel of input array
    :param test_x: A 3-D numpy array in the format (samples,channels,timesteps)
    :return: (Means, Stds): numpy arrays of dimensions (channels,1)

    """
    means = np.zeros(test_x.shape[1], dtype=np.float32)
    stnds = np.zeros(test_x.shape[1], dtype=np.float32)
    for i in range(test_x.shape[1]):
        means[i] = test_x[:, i, :].mean()
        stnds[i] = test_x[:, i, :].std()
    return means.reshape((-1, 1)), stnds.reshape((-1, 1))


def ckd_to_kernels(ckd, candidate_kernels, candidate_dilations):
    """
    :param ckd: A channel-kernel-dilation 2d array of dimensions (n_kernels,3)
    :param candidate_kernels: The set of base kernels used by LightWaveS
    :param candidate_dilations: The set of base dilations used by LightWaveS
    :return: Tuple of kernels in format suitable for the core algorithm (similar to ROCKET)
    """
    num_channel_indices = np.ones(ckd.shape[0], dtype=np.int32)
    channel_indices = ckd[:, 0]
    biases = np.zeros_like(num_channel_indices, dtype=np.float32)
    dilations = 2 ** candidate_dilations[ckd[:, 2]].flatten().astype(np.int32)
    lengths = np.array([len(candidate_kernels[i]) for i in ckd[:, 1]], dtype=np.int32)
    paddings = np.multiply((lengths - 1), dilations) // 2
    weights = candidate_kernels[ckd[:, 1]].flatten().astype(np.float32)

    return (
        weights,
        lengths,
        biases,
        dilations,
        paddings,
        num_channel_indices,
        channel_indices,
    )


def get_fixed_candidate_kernels():
    """
    :return: The set of base kernels used by LightWaveS (same as that of MINIROCKET)
    """
    kernel_set = np.array([np.array(p) for p in multiset_permutations(([2] * 3 + [-1] * 6))], dtype=np.float32)
    return kernel_set


def measure_transform_times():
    """
    Loads the datasets and corresponding LightWaveS model and measures the time required for scaling + kernel applications to a random test sample. The measured times are saved to a csv file.
    """
    global_times = []
    set_num_threads(multiprocessing.cpu_count())
    for filename in sorted(glob.glob(F"Industrial/*.npz")):
        local_times = []
        dataset = filename.split("/")[-1].split(".")[0]
        data = np.load(filename)
        test_x = data['test_x'].astype(np.float32)
        ## In reality, the means and standard deviations of the training set would be required, but we are interested in the standard scaling time and not the actual result, so we just use the equivalent test set quantities
        m, s = GetScaleParams(test_x)

        candidate_kernels = get_fixed_candidate_kernels()
        dilations = np.arange(0, np.log2(32) + 1).astype(np.int32)

        with open(F"L2_{dataset}_matrix.pickle", "rb") as in_f:
            kernel_matrix_final, feat_mask = pickle.load(in_f)

        kernels = ckd_to_kernels(kernel_matrix_final, candidate_kernels, dilations)

        for i in range(100):
            print(f'{dataset} - Rep: {i}')
            ind = np.random.choice(test_x.shape[0])
            gc.collect()
            start = timer()
            t = (test_x[ind:ind + 1] - m) / s
            feats = _apply_2layer_kernels(t, kernels)[:, feat_mask]
            end = timer()
            local_times.append(end - start)
        global_times.append(local_times)
    np.savetxt(F"jetson_proposed_transform_times_l2.csv", np.array(global_times), delimiter=',')


if __name__ == "__main__":
    measure_transform_times()
