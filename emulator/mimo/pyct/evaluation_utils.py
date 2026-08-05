import torch
import numpy as np
from functools import partial
from generate_training_dataset import map2tiles, tiles2map

import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.calibration import calibration_curve

def rmse(y_hat, y):
    if isinstance(y_hat, torch.Tensor):
        return torch.mean(torch.sqrt((y_hat-y)**2))
    return np.mean(np.sqrt((y_hat-y)**2))


#making these 2 different functions as to not incur extra computation when computing loss for training
#Could use default threshold as with difference_map
def rmse_cloud_only(y_hat, y, threshold):
    inds = np.where((y_hat >= threshold) | (y >= threshold))

    return rmse(y_hat[inds], y[inds])


def linear_fit_stats(y_true, y_pred):
     if len(y_true) < 2:
         return {
             "slope": np.nan,
             "intercept": np.nan,
             "r": np.nan,
             "rmse": np.nan,
             "bias": np.nan,
         }
     slope, intercept, r, _, _ = stats.linregress(y_true, y_pred)
     return {
         "slope": slope,
         "intercept": intercept,
         "r": r,
         "rmse": rmse(y_true, y_pred),
         "bias": np.mean(y_pred - y_true),
     }


def qcut_or_cut(values, n_bins=10, logspace=False, eps=1e-8):
    values = np.asarray(values)
    if logspace:
        v = np.maximum(values, eps)
        edges = np.geomspace(v.min(), v.max(), n_bins + 1)
        return pd.cut(v, bins=edges, include_lowest=True, duplicates="drop")
    return pd.qcut(values, q=n_bins, duplicates="drop")
 


def tiled_rmse(y_hat, y, patch_size, equal_size_output=True, eval_func=rmse):
    y_hat_tiled = map2tiles(np.squeeze(y_hat), patch_size, patch_size)
    y_tiled = map2tiles(np.squeeze(y), patch_size, patch_size)
    print(y_hat.shape, y.shape, y_tiled.shape)

    if equal_size_output:
        output = np.zeros(y_hat_tiled.shape)
    else:
        output = np.zeros(y_hat_tiled.shape[0])

    for i in range(y_hat_tiled.shape[0]):
        mean_rmse = rmse(y_hat_tiled[i], y_tiled[i]).mean()
        output[i] = mean_rmse

    if equal_size_output:
        output = tiles2map(output, y_hat.shape[1], y_hat.shape[2], patch_size)

    return output

def tiled_rmse_cloud_only(y_hat, y, patch_size, threshold, equal_size_output=True, ):

    eval_func = partial(rmse_cloud_only, threshold=threshold)

    return tiled_rmse(y_hat, y, patch_size, equal_size_output, eval_func)



def difference_map(y_hat, y, threshold = np.inf):

    if np.isfinite(threshold):
        output = np.zeros(y_hat.shape)
        inds = np.where((y >= threshold) | (y_hat >= threshold))
        output[inds] = y_hat[inds] - y[inds]
        return output

    return y_hat - y


def tiled_difference_map(y_hat, y, patch_size, threshold=np.inf):

    y_hat_tiled = map2tiles(np.squeeze(y_hat), patch_size, patch_size)
    y_tiled = map2tiles(np.squeeze(y), patch_size, patch_size)
 
    output = np.zeros(y_hat_tiled.shape) 

    for i in range(y_hat_tiled.shape[0]):
        mean_diff = difference_map(y_hat_tiled[i], y_tiled[i], threshold).mean()
        output[i] = mean_diff

    print("tiled_diff_map", y_hat.shape, y.shape, y_hat_tiled.shape, patch_size)
    output = tiles2map(output, y_hat.shape[1], y_hat.shape[2], patch_size)


    return output



