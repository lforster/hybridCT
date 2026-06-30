import os
import numpy as np
import xarray as xr
import argparse
import yaml
from joblib import dump, load
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def save_scaler(fname, scaler):
    if scaler is not None:
        with open(fname, "wb") as f:
            dump(scaler, f, True, pickle.HIGHEST_PROTOCOL)

def scaler_fit(scaler_, dataset, save_fname=None):
    n_samples, n_x, n_y, n_features = dataset.shape
    scaler = scaler_.fit(dataset.reshape(n_samples*n_x*n_y, n_features))
    if save_fname:
        save_scaler(save_fname, scaler)
    return scaler

def scaler_transform(scaler, dataset, method):
    shape = dataset.shape
    n_samples, n_x, n_y, n_features = shape
    dataset_scaled = scaler.__getattribute__(method)(dataset.reshape(n_samples*n_x*n_y,n_features)).reshape(shape)
    return dataset_scaled

def create_dataset_filename(config, file_ind=0):
    tag = os.path.splitext(os.path.basename(config['input_fname'][file_ind]))[0] # dataset identifier tag for filename
    res_tag = int(config['resolution']*1e3) # resolution tag in [m] for filename
    save_fname = f"training_data_{tag}_res{res_tag:d}m_tiles{config['tile_size']}x{config['tile_size']}_stride{config['stride']}_offset{config['offset_x']}x{config['offset_y']}.nc"
    return save_fname

def read_preprocessed_tiles(config, tile_fname=None):
    if tile_fname is None:
        tile_fname = os.path.join(config['out_dir'], create_dataset_filename(config))
    print(f"Reading preprocessed tiles from {tile_fname}")
    with xr.open_dataset(tile_fname) as data:
        return data[config["input_field"]].data, data[config["target_field"]].data, data.nx, data.ny

def save2dataset(X, y, nx, ny, config, tile_fname=None):
    if tile_fname is None:
        tile_fname = os.path.join(config['out_dir'], create_dataset_filename(config))
    attributes = ["grid_field", "vza_field", "input_field", "target_field", "target_fname", "stride", "offset_x", "offset_y", "tile_size", "resolution", "dx"]
    attrs = {k:config[k] for k in attributes}
    attrs["nx"] = nx
    attrs["ny"] = ny
    ds = xr.Dataset({
        config["input_field"]: (["samples","tx","ty","nviews"], X),
        config["target_field"]: (["samples","tx","ty","nviews"], y),
        }, attrs=attrs)
    if "angles" in config:
        ds["angles"] = (["nviews"], config["angles"])
    print(f"Saving preprocessed tiles to {tile_fname}")
    ds.to_netcdf(tile_fname)


def check_if_cloudy(y, config):

    inds = []
    if config["stratify_check_center"]:
        center = [int(round(y.shape[0]/2)), int(round(y.shape[1]/2))]
        step = int(round(config["stratify_center_window_size"] / 2))
        area = y[center[0]-step:center[0]+step,center[1]-step:center[1]+step]

        inds = np.where(area >= config["tau_min_threshold"])
    else:
        inds = np.where(y >= config["tau_min_threshold"])

    return ((len(inds[0]) / y.shape[0]*y.shape[1]) >= config["cloudy_tile_percent_threshold"])


def stratified_train_test_val_split(x, y, config):

    dataset_size = x.shape[0]
    num_test_exs = round(config["test_ratio"] * dataset_size)
    num_val_exs = round(config["val_ratio"] * dataset_size)
    counts = []
    type_inds = []
    test_indices = []
    val_indices = []
    train_indices = []

    for i in range(x.shape[0]):
        if check_if_cloudy(y[i], config):
            type_inds.append(i)

    percentage_of_dataset = len(type_inds) / dataset_size
    # calculate how many test & val exaples to take from the given water type
    num_test_exs_of_type = round(percentage_of_dataset * num_test_exs)
    num_val_exs_of_type = round(percentage_of_dataset * num_val_exs)

    # randomly sample examples from the given water type
    test_val_indices = np.random.choice(type_inds, size=num_test_exs_of_type + num_val_exs_of_type, replace=False)
    test_indices.extend(test_val_indices[0:num_test_exs_of_type])
    val_indices.extend(test_val_indices[num_test_exs_of_type:])

    test_val = test_indices + val_indices
    train_indices = np.setdiff1d(np.arange(dataset_size), test_val).tolist()

    # ensure there isn't overlap between test and val splits
    all_indices = train_indices + test_indices + val_indices
    unique_indices, counts = np.unique(all_indices, return_counts=True)
    dup = unique_indices[counts > 1]
    if len(dup) > 0:
        raise Exception("Train, test, and val splits overlap. Redefine splits.")

    # subset based on indices
    X_train = x[train_indices]
    X_test = x[test_indices]
    X_val = x[val_indices] 
    y_train = y[train_indices]
    y_test = y[test_indices]
    y_val = y[val_indices]

    return X_train, X_test, X_val, y_train, y_test, y_val


def log_unscale_data_sub(_data, fill_value=None):
    #TODO zero_inds have to be stored.
    # need them for y_hat, since fill_value might have changed
    if fill_value is not None:
        zero_inds = np.where(_data == fill_value)
    else:
        zero_inds = np.where(_data <= np.nanmin(data))
    data = np.exp(_data)
    data[zero_inds] = 0.0
    return data

def log_unscale_data(_data, per_channel=False, fill_value=np.inf):
    data = _data.copy()
    if per_channel:
        for c in range(data.shape[1]):
            if not np.isfinite(fill_value):
                fill_value = np.nanmin(data[:,c,:,:])
            data[:,c,:,:] = log_unscale_data_sub(data[:,c,:,:], fill_value)
    else:
        if not np.isfinite(fill_value):
            fill_value = np.nanmin(data)
        data = log_unscale_data_sub(data, fill_value)
    return data

def log_scale_data_sub(_data, fill_value):
    data = np.log(_data)
    data[_data==0] = fill_value #Giving 0s unique (and very small) value in log space
    return data

def log_scale_data(_data, per_channel=False, fill_value=None):
    data = _data.copy()
    if per_channel:
        for c in range(_data.shape[1]):
            data[:,c,:,:] = log_scale_data_sub(_data[:,c,:,:], fill_value)
    else:
        data = log_scale_data_sub(_data, fill_value)

    return data


def load_train_test_val_dataset(out_dir):
    X_train = np.load(os.path.join(out_dir, "x_train.npy"), allow_pickle=True) #.transpose(0,3,1,2)
    y_train = np.load(os.path.join(out_dir, "y_train.npy"), allow_pickle=True) #.transpose(0,3,1,2)
    X_test = np.load(os.path.join(out_dir, "x_test.npy"), allow_pickle=True) #.transpose(0,3,1,2)
    y_test = np.load(os.path.join(out_dir, "y_test.npy"), allow_pickle=True) #.transpose(0,3,1,2)
    X_val = np.load(os.path.join(out_dir, "x_val.npy"), allow_pickle=True) #.transpose(0,3,1,2)
    y_val = np.load(os.path.join(out_dir, "y_val.npy"), allow_pickle=True) #.transpose(0,3,1,2)
    return X_train, X_test, X_val, y_train, y_test, y_val


def preprocess_training_data(rad_tiled, target_tiled, config):
    if config["stratify"]:
        #split tiles in stratified way
        X_train, X_test, X_val, y_train, y_test, y_val = \
            stratified_train_test_val_split(rad_tiled, target_tiled, config)
    else:
        all_indices = list(range(rad_tiled.shape[0]))
        train_val_ind, test_ind = train_test_split(all_indices, test_size=config["test_ratio"])
        train_ind, val_ind = train_test_split(train_val_ind, test_size=config["val_ratio"])

        X_train = rad_tiled[train_ind]
        y_train = target_tiled[train_ind]
        X_test = rad_tiled[test_ind]
        y_test = target_tiled[test_ind]
        X_val = rad_tiled[val_ind]
        y_val = target_tiled[val_ind]

    out_dir = config["out_dir"]
    np.save(os.path.join(out_dir, "x_train"), X_train)
    np.save(os.path.join(out_dir, "x_test"),  X_test)
    np.save(os.path.join(out_dir, "x_val"),  X_val)
    np.save(os.path.join(out_dir, "y_train"),  y_train)
    np.save(os.path.join(out_dir, "y_test"),  y_test)
    np.save(os.path.join(out_dir, "y_val"),  y_val)
    return X_train, X_test, X_val, y_train, y_test, y_val


def scale_training_data(input_tiles, target_tiles, config):
    # Log-scale
    print(target_tiles.mean(), input_tiles.mean())
    if config["log_scale_input"]:
        input_tiles = log_scale_data(input_tiles, True, config['fill_value'])
    if config["log_scale_target"]:
        target_tiles = log_scale_data(target_tiles, True, config['fill_value'])

    print(target_tiles.mean(), input_tiles.mean())

    #Normalize
    input_scaler = None
    output_scaler = None
    out_dir = config["out_dir"]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    inp_scaler_fname = os.path.join(out_dir, "input_scaler.pkl")
    if config["input_scaler"] == "Standard":
        input_scaler = scaler_fit(StandardScaler(), input_tiles, save_fname=inp_scaler_fname)
        input_tiles = scaler_transform(input_scaler, input_tiles, method="transform")
    elif config["input_scaler"] == "MinMax":
        input_scaler = scaler_fit(MinMaxScaler(), input_tiles, save_fname=inp_scaler_fname)
        input_tiles = scaler_transform(input_scaler, input_tiles, method="transform")

    out_scaler_fname = os.path.join(out_dir, "output_scaler.pkl")
    if config["target_scaler"] == "Standard":
        target_scaler = scaler_fit(StandardScaler(), target_tiles, save_fname=out_scaler_fname)
        target_tiles = scaler_transform(target_scaler, target_tiles, method="transform")
    elif config["target_scaler"] == "MinMax":
        target_scaler = scaler_fit(MinMaxScaler(), target_tiles, save_fname=out_scaler_fname)
        target_tiles = scaler_transform(target_scaler, target_tiles, method="transform")
 
    print(target_tiles.mean(), input_tiles.mean())
    return input_tiles, target_tiles


def load_scalers(config):

    model_dir = config["out_dir"]
    if "model_dir" in config:
        model_dir = config["model_dir"]

    input_scaler = None
    scaler_fname = os.path.join(model_dir, "input_scaler.pkl")
    try:
        input_scaler = load(scaler_fname)
    except FileNotFoundError:
        print(f"Error: The file at '{scaler_fname}' does not exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    output_scaler = None
    scaler_fname = os.path.join(model_dir, "output_scaler.pkl")
    try:
        output_scaler = load(scaler_fname)
    except FileNotFoundError:
        print(f"Error: The file at '{scaler_fname}' does not exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return input_scaler, output_scaler


def read_yaml(fpath_yaml):
    yml_conf = None
    with open(fpath_yaml) as f_yaml:
        yml_conf = yaml.safe_load(f_yaml)
    return yml_conf


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    config = read_yaml(args.yaml)
    rad_tiled, target_tiled, nx, ny = read_preprocessed_tiles(config)
    input_tiles_scaled, target_tiles_scaled = scale_training_data(rad_tiled, target_tiled, config)
    preprocess_training_data(input_tiles_scaled, target_tiles_scaled, config)


