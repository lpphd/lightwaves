import pandas as pd
import numpy as np
import glob


def prepare_mafaulda():
    """
    Read the raw csv files of the MAFAULDA dataset and prepare a sub-sampled version, as well as an npz file with a train-test split of 85%-15%.
    """
    conditions = ['normal', 'imbalance', 'horizontal-misalignment', 'vertical-misalignment', 'overhang', 'underhang']
    data = []
    labels = []
    for i in range(len(conditions)):
        filenames = sorted(glob.glob(F"../full/{conditions[i]}/*.csv")) + sorted(
            glob.glob(F"../full/{conditions[i]}/*/*.csv")) + sorted(glob.glob(F"../full/{conditions[i]}/*/*/*.csv"))
        samples = 0
        for filename in filenames:
            df = pd.read_csv(filename, header=None).values[::250, :].reshape((1, 8, -1))
            data.append(df)
            samples += 1
        labels.append(np.ones(samples) * i)
    x = np.concatenate(data)
    y = np.concatenate(labels)
    np.savez(F"../Datasets/MAFAULDA/mafaulda_orig.npz", x=x, y=y)

    np.random.seed(42)
    mask = np.zeros(x.shape[0], dtype=bool)
    idces = np.random.choice(x.shape[0], int(x.shape[0] * 0.85), replace=False)
    mask[idces] = True
    train_x = x[np.where(mask)]
    train_y = y[np.where(mask)]
    test_x = x[np.where(~mask)]
    test_y = y[np.where(~mask)]
    np.savez(F"../Datasets/MAFAULDA/mafaulda.npz", train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)


def prepare_turbofan():
    """
        Read the raw text files of the TURBOFAN dataset and prepare a binary classification problem with the class being more (0) or fewer (1) than 20 operating cycles remaining.
        Since the original data can consist of different length time series, we first find a minimum common length of the test samples to turn the problem into classification of equal length samples.
        The training samples are adjusted accordingly based on this length.
        """
    np.random.seed(42)
    for serial in range(1, 5):
        train_ds = pd.read_csv(F"../Datasets/CMAPSSData/train_FD00{serial}.txt", header=None, delimiter=" ").values[:,
                   :-2]
        test_ds = pd.read_csv(F"../Datasets/CMAPSSData/test_FD00{serial}.txt", header=None, delimiter=" ")
        min_common_test_len = test_ds.groupby(0).agg({1: max}).min().item()
        test_ds = test_ds.values[:, :-2]
        test_ds_labels = pd.read_csv(F"../Datasets/CMAPSSData/RUL_FD00{serial}.txt", header=None).values.reshape((-1,))
        train_data = []
        train_labels = []
        for n in range(int(max(train_ds[:, 0]))):
            machine_data = train_ds[train_ds[:, 0] == n + 1]
            max_cycles = max(machine_data[:, 1])
            for i in range(machine_data.shape[0] - min_common_test_len):
                train_data.append(machine_data[i:i + min_common_test_len, :].reshape((1, -1, min_common_test_len)))
                train_labels.append(int(max_cycles - machine_data[i + min_common_test_len, 1]))
        train_x = np.concatenate(train_data, axis=0)
        train_y = np.array(train_labels)
        train_x_pos = train_x[train_y < 20]
        train_x_neg = train_x[train_y >= 20]
        neg_indices = np.random.choice(train_x_neg.shape[0], size=train_x_pos.shape[0], replace=False)
        train_x_neg = train_x_neg[neg_indices]
        train_y_pos = np.ones_like(train_y[train_y < 20])
        train_y_neg = np.zeros_like(train_y[train_y >= 20][neg_indices])
        train_x = np.concatenate((train_x_pos, train_x_neg), axis=0)
        train_y = np.concatenate((train_y_pos, train_y_neg), axis=0)
        test_data = []
        test_labels = []
        for n in range(int(max(test_ds[:, 0]))):
            machine_data = test_ds[test_ds[:, 0] == n + 1]
            max_cycles = max(machine_data[:, 1]) + test_ds_labels[n]
            for i in range(machine_data.shape[0] - min_common_test_len):
                test_data.append(machine_data[i:i + min_common_test_len, :].reshape((1, -1, min_common_test_len)))
                test_labels.append(int(max_cycles - machine_data[i + min_common_test_len, 1]))
        test_x = np.concatenate(test_data, axis=0)
        test_y = np.array(test_labels)
        test_x_pos = test_x[test_y < 20]
        test_x_neg = test_x[test_y >= 20]
        neg_indices = np.random.choice(test_x_neg.shape[0], size=test_x_pos.shape[0], replace=False)
        test_x_neg = test_x_neg[neg_indices]
        test_y_pos = np.ones_like(test_y[test_y < 20])
        test_y_neg = np.zeros_like(test_y[test_y >= 20][neg_indices])
        test_x = np.concatenate((test_x_pos, test_x_neg), axis=0)
        test_y = np.concatenate((test_y_pos, test_y_neg), axis=0)
        np.savez(F"../Datasets/CMAPSSData/FD00{serial}.npz", train_x=train_x, train_y=train_y, test_x=test_x,
                 test_y=test_y)
