from numba import njit, prange
from sklearn.feature_selection import VarianceThreshold, f_classif
import numpy as np
from scipy.stats import rankdata


def ScalePerChannel(train_x, test_x):
    """
    Applies standard scaling (across channels) on train and test arrays
    :param train_x: Train set array of dimension (samples,channels,timesteps)
    :param test_x: Test set array of dimension (samples,channels,timesteps)
    :return: (train_x_scaled, test_x_scaled) : Standard scaled (across channels) arrays
    """
    out_train_x = np.zeros_like(train_x, dtype=np.float32)
    out_test_x = np.zeros_like(test_x, dtype=np.float32)
    for i in range(train_x.shape[1]):
        out_train_x[:, i, :] = (train_x[:, i, :] - train_x[:, i, :].mean()) / (np.std(train_x[:, i, :]) + 1e-8)
        out_test_x[:, i, :] = (test_x[:, i, :] - train_x[:, i, :].mean()) / (np.std(train_x[:, i, :]) + 1e-8)

    return out_train_x, out_test_x


def ScalePerChannelTrain(train_x):
    """
    Applies standard scaling (across channels) on train array
    :param train_x: Train set array of dimension (samples,channels,timesteps)
    :return: train_x_scaled: Standard scaled (across channels) array
    """
    out_train_x = np.zeros_like(train_x, dtype=np.float32)
    for i in range(train_x.shape[1]):
        out_train_x[:, i, :] = (train_x[:, i, :] - train_x[:, i, :].mean()) / (np.std(train_x[:, i, :]) + 1e-8)

    return out_train_x


@njit(
    fastmath=True,
    cache=True,
)
def vector_pearson_corr(X, y):
    """
    Returns Pearson correlation using Numba for X,y vectors
    """
    X_diff = X - X.mean()
    y_diff = y - y.mean()
    num = (X_diff * y_diff).sum()
    den = np.sqrt((X_diff ** 2).sum() * (y_diff ** 2).sum())
    return num / den


@njit(
    "float32[:](float32[:,:],float32[:,:])",
    parallel=True,
    fastmath=True
)
def pearson_corr_numba(X, y):
    """
        Returns Pearson correlation using Numba for X,y arrays
    """
    res = np.zeros((X.shape[1], 1), dtype=np.float32)
    for i in prange(X.shape[1]):
        res[i] = vector_pearson_corr(X[:, i], y[:, 0])
    return res[:, 0]


def pearson_corr(X, y):
    """
        Returns Pearson correlation for X,y arrays
    """
    X_diff = X - X.mean(axis=0)
    y_diff = y - y.mean(axis=0)
    num = (X_diff * y_diff).sum(axis=0)
    den = np.sqrt((X_diff ** 2).sum(axis=0) * (y_diff ** 2).sum(axis=0))
    return num / den


def spearman_corr(ranked_X, unranked_y):
    """
        Returns Spearman correlation using Numba for X,y arrays
    :param ranked_X: Ranked version of X matrix (feature matrix)
    :param unranked_y: Unranked version of y matrix (class target matrix)
    """
    y_r = rankdata(unranked_y, axis=0).astype(np.float32)
    return pearson_corr_numba(ranked_X, y_r)


def mrmr_feature_selection(X, y, K=100):
    """
    Returns the top k features using minimum minimum redundancy - maximum relevance method.
    :param X: Features array
    :param y: Classes array
    :param K: Number of top features to return
    :return: (idces,out_scores,orig_scores): Indices of selected features, adjusted scores based on mrmr, original correlation scores based on Pearson correlation
    """
    try:
        var_mask = VarianceThreshold().fit(X).get_support()
    except ValueError as e:
        print("Exception: ", e)
        var_mask = np.ones(X.shape[1], dtype=np.bool)
    X_v = X[:, var_mask]
    or_idx = np.zeros(X.shape[1], dtype=np.bool)
    scores = np.nan_to_num(f_classif(X_v, y)[0], posinf=0, neginf=0).astype(np.float32)
    selected_features_mask = np.zeros_like(X_v[0, :], dtype=np.bool)
    first_idx = np.argmax(scores)
    selected_features_mask[first_idx] = True
    corr = pearson_corr_numba(X_v, X_v[:, first_idx:first_idx + 1])
    corr_sum = np.abs(corr)
    num_feats = min(K, var_mask.sum())
    out_scores = np.zeros(num_feats, dtype=np.float32)
    original_scores = np.zeros(num_feats, dtype=np.float32)
    transl_idces = np.zeros(num_feats, dtype=np.int32)
    out_scores[0] = scores[first_idx]
    original_scores[0] = scores[first_idx]
    transl_idces[0] = first_idx
    for i in range(num_feats - 1):
        corr_mean = corr_sum / (i + 1)
        adj_scores = np.divide(scores * ~selected_features_mask, corr_mean)
        new_idx = np.argmax(adj_scores)
        transl_idces[i + 1] = new_idx
        out_scores[i + 1] = adj_scores[new_idx]
        original_scores[i + 1] = scores[new_idx]
        selected_features_mask[new_idx] = True
        corr = pearson_corr_numba(X_v, X_v[:, new_idx:new_idx + 1])
        corr_sum += np.abs(corr)
    or_idx[np.where(var_mask)[0][selected_features_mask]] = True
    sorted_indices = np.argsort(transl_idces)
    idces = np.where(or_idx)[0].astype(np.int32)
    out_scores = out_scores[sorted_indices]
    original_scores = original_scores[sorted_indices]
    return idces, out_scores, original_scores


def anova_feature_selection(X, y, N=100):
    """
    Returns the top N features using ANOVA method.
    :param X: Features array
    :param y: Classes array
    :param N: Number of top features to return
    :return: (idces,scores): Indices of selected features, ascores based on ANOVA
    """
    var_mask = VarianceThreshold().fit(X).get_support()
    or_idx = np.zeros(X.shape[1], dtype=np.bool)
    X_v = np.round(X[:, var_mask].copy(), 7)  # Quick fix for weird numerical precision issue
    scores = np.nan_to_num(f_classif(X_v, y)[0], posinf=0, neginf=0)
    idces = np.argsort(scores)[::-1][:N]
    scores = scores[idces]

    scores = scores[np.argsort(idces)].astype(np.float32)
    idces = np.sort(idces)
    or_idx[np.where(var_mask)[0][idces]] = True
    idces = np.where(or_idx)[0].astype(np.int32)
    return idces, scores
