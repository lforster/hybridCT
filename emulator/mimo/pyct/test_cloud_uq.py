from argparse import ArgumentParser
from typing import List, Tuple
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import scipy.stats
import numpy as np
import pandas as pd
from pathlib import Path
import os
import logging
import multiprocessing as mp

from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from mimo.models.ensemble import EnsembleModule
#from mimo.models.evidential_unet import EvidentialUnetModel
#from mimo.models.utils import repeat_subnetworks
from cloud_datamodule import CloudDataModule
import utils

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def make_predictions(model, dataset, device: str, batch_size: int = 5, epsilon: float = 0.0):
    inputs = []
    y_preds = []
    y_trues = []
    aleatoric_vars = []
    epistemic_vars = []
    log_params = []
    
    loader = DataLoader(dataset, batch_size=batch_size)

    for data in tqdm(loader):
        images = data['image'].to(device)
        labels = data['label'].cpu()

        labels = labels.unsqueeze(1)
        labels = labels.repeat(1, model.num_subnetworks, 1, 1, 1)

        #image = repeat_subnetworks(images, num_subnetworks=self.num_subnetworks)
        #label = repeat_subnetworks(labels, num_subnetworks=self.num_subnetworks)


        images.requires_grad = True
        labels.requires_grad = True

        y_pred, log_param = model(images)

        loss = model.loss_fn(y_pred, log_param, labels)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = images.grad.data

        # Call FGSM Attack
        #perturbed_data = fgsm_attack(images, epsilon, data_grad)

        # Predict on the perturbed image
        #y_pred, log_param = model(perturbed_data)

        y_pred = y_pred.cpu().detach()
        log_param = log_param.cpu().detach()
        y_true = data['label'].cpu().detach()

        inputs.append(images.cpu().detach()) #perturbed_data.cpu().detach())
        y_preds.append(y_pred)
        y_trues.append(y_true)
        log_params.append(log_param)
        #mean, aleatoric_var, epistemic_var = model(images)
        #inputs.append(images.cpu().detach())
        #y_trues.append(data['label'].cpu().detach())
        #y_preds.append(mean.cpu().detach())
        #aleatoric_vars.append(aleatoric_var.cpu().detach())
        #epistemic_vars.append(epistemic_var.cpu().detach())
 
    inputs = torch.cat(inputs, dim=0).detach()
    y_preds = torch.cat(y_preds, dim=0).detach() #.clip(min=0).detach()
    y_trues = torch.cat(y_trues, dim=0).detach() #.clip(min=0).detach()
    log_params = torch.cat(log_params, dim=0).detach()
    #aleatoric_vars = torch.cat(aleatoric_vars, dim=0).detach()
    #epistemic_vars = torch.cat(epistemic_vars, dim=0).detach()

    aleatoric_vars, epistemic_vars = compute_uncertainties(
        model.loss_fn,
        y_preds=y_preds,
        log_params=log_params,
    )

    y_preds = torch.squeeze(y_preds.mean(dim=1))

    print(y_trues.shape, aleatoric_vars.shape, epistemic_vars.shape, inputs.shape, y_preds.shape)

    return (
        inputs,
        y_preds, #[:,4], #.mean(axis=1)[:, 0], 
        y_trues, #[:, 4], 
        aleatoric_vars, #[:, 4], 
        epistemic_vars, #[:, 4],
        aleatoric_vars + epistemic_vars #[:, 4] + epistemic_vars[:, 4],
    )


def convert_to_pandas(y_preds, y_trues, aleatoric_vars, epistemic_vars, combined_vars):
    print(y_preds.numpy().flatten().shape, y_trues.numpy().flatten().shape, np.sqrt(aleatoric_vars.numpy()).flatten().shape, np.sqrt(epistemic_vars.numpy()).flatten().shape, np.sqrt(combined_vars.numpy()).flatten().shape)
    print(y_preds.numpy().shape, y_trues.numpy().shape, np.sqrt(aleatoric_vars.numpy()).shape, np.sqrt(epistemic_vars.numpy()).shape, np.sqrt(combined_vars.numpy()).shape)
    data = np.stack([
        y_preds.numpy().flatten(),
        y_trues.numpy().flatten(), 
        np.sqrt(aleatoric_vars.numpy()).flatten(),
        np.sqrt(epistemic_vars.numpy()).flatten(),  
        np.sqrt(combined_vars.numpy()).flatten(),  
    ], axis=0).T
    
    df = pd.DataFrame(
        data=data,
        columns=['y_pred', 'y_true', 'aleatoric_std', 'epistemic_std', 'combined_std']
    )
    return df


def compute_uncertainties(criterion, y_preds, log_params):
    """
    Args:
        y_preds: [B, S, C, H, W]
    """
    print("HERE UNCERT", y_preds.shape)
    _, S, _, _, _ = y_preds.shape
   
    print("HERE UNCERT2", y_preds.min(), y_preds.mean(), y_preds.max(), y_preds.std(), log_params.min(), log_params.mean(), log_params.max(), log_params.std())

    stds = criterion.std(y_preds, log_params)
    aleatoric_variance = torch.square(stds).mean(dim=1)
    
    print(aleatoric_variance.mean(), aleatoric_variance.std(), aleatoric_variance.shape)

    if S > 1:
        y_preds_mean = y_preds.mean(dim=1, keepdims=True)
        epistemic_variance = torch.square(y_preds - y_preds_mean).sum(dim=1) / (S - 1)
    else:
        epistemic_variance = torch.zeros_like(aleatoric_variance)
        
    return aleatoric_variance, epistemic_variance


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df['error'] = np.abs(df['y_pred'] - df['y_true'])
    return df


def create_precision_recall_plot(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(by='combined_std', ascending=False)
    
    percentiles = np.arange(100)/100.
    cutoff_inds = (percentiles * df.shape[0]).astype(int)
    
    mae = [df.iloc[cutoff:]["error"].mean() for cutoff in tqdm(cutoff_inds)]
    mse = [np.square(df.iloc[cutoff:]["error"]).mean() for cutoff in tqdm(cutoff_inds)]
    
    df_cutoff = pd.DataFrame({'percentile': percentiles, 'mae': mae, 'rmse': np.sqrt(mse)})
    
    return df_cutoff


def compute_ppf(params):
    p, y_pred, aleatoric_std, distribution = params
    return distribution.ppf(p, loc=y_pred, scale=aleatoric_std / np.sqrt(2))
    
def create_calibration_plot(df: pd.DataFrame, distribution, processes) -> pd.DataFrame:
    
    y_true = df['y_true'].to_numpy()
    y_pred = df['y_pred'].to_numpy()
    print(y_true.mean(), y_true.std(), y_pred.mean(), y_pred.std())
    aleatoric_std = df['aleatoric_std'].to_numpy()

    expected_p = np.arange(7) / 6.

    print(aleatoric_std, aleatoric_std / np.sqrt(2), y_pred.mean(), y_pred.std(), y_pred.min(), y_pred.max())

    print('- computing ppfs')
    with mp.Pool(processes=processes) as pool:
        params = [(p, y_pred, aleatoric_std, distribution) for p in expected_p]
        results = pool.imap(compute_ppf, params, chunksize=1)
        ppfs = np.array(list(tqdm(results, total=len(expected_p))))

    print(ppfs)
    print('- computing observed_p')
    below = y_true[None, :] < ppfs
    observed_p = below.mean(axis=1)
    
    df_calibration = pd.DataFrame({'Expected Conf.': expected_p, 'Observed Conf.': observed_p})
    return df_calibration


def expected_calibration_error(calibration_df: pd.DataFrame) -> float:
     r"""
     Compute the expected calibration error.
 
     $$ ECE = \sum_{i=1}^{M} w_i \left|p_{obs} - p_{exp}\right| $$,
  
     where $w_i = \frac{N_i}{N}$ if `count` is present in the calibration dataframe, and $w_i = 1/M$ otherwise.
  
     Args:
          calibration_df (pd.DataFrame): The calibration dataframe.

     Returns:
          `float`: The expected calibration error.
     """
  
     abs_diff = (calibration_df['Observed Conf.'] - calibration_df['Expected Conf.']).abs()
  
     if 'count' in calibration_df.columns:
          total_count = calibration_df['count'].sum()
          weight = calibration_df['count'] / total_count
          return (abs_diff * weight).sum()
  
     return  abs_diff.mean()

def plot_calibration(
      df: pd.DataFrame,
      label: Optional[str] = None,
      show_ece: bool = True,
      show_ideal: bool = True,
      show_area: bool = True,
      ax: Optional[plt.Axes] = None,
  ) -> plt.Axes:
      """
      Plot the calibration curve for a model.
  
      Args:
          df: DataFrame containing the calibration data.
          label: Label for the calibration curve.
          show_ece: Whether to display the Expected Calibration Error (ECE) in the plot.
          show_ideal: Whether to display the ideal calibration line.
          show_area: Whether to color the area between the calibration curve and the ideal line.
          ax: Matplotlib axes to plot on.
  
      Returns:
          Matplotlib axes with the calibration plot.
      """
  
      # Set the style of seaborn for more attractive plots
      sns.set_theme(style="ticks")
  
      if ax is None:
          fig, ax = plt.subplots(figsize=(5, 5))
  
      # Adding the calibration line from dataframe with marker options
      sns.lineplot(x='Expected Conf.', y='Observed Conf.', data=df, ax=ax, label=label, linewidth=1.5)
  
  
      if show_ideal:
          # Adding a line plot for perfectly calibrated predictions
          ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated', linewidth=1.5)
  
      if show_area:
          # Coloring area between the calibration line and the perfect calibration line
          ax.fill_between(df['Expected Conf.'], df['Expected Conf.'], df['Observed Conf.'], alpha=0.2)
  
      # Adding the Expected Calibration Error (ECE) value as a text in the plot in bottom right corner
      if show_ece:
          ece = expected_calibration_error(df)
          ax.text(1, 0.05, f'ECE: {ece:.3f}', ha='right', va='center', transform=ax.transAxes)
  
      # Setting labels with increased font size for better readability
      ax.set_xlabel('Expected Proportion', fontsize=12)
      ax.set_ylabel('Observed Proportion', fontsize=12)
  
      # Set the range of x and y axes
      ax.set_xlim([0, 1])
      ax.set_ylim([0, 1])
  
      ax.legend(frameon=False)
  
      # Additional customization: removing the top and right spines for a cleaner look
      sns.despine()
  
      return ax

def plot_precision_recall(
      df: pd.DataFrame,
      metric: str = 'rmse',
      label: Optional[str] = None,
      ax: Optional[plt.Axes] = None,
  ) -> plt.Axes:
      """
      Plot the precision-recall curve for a model.
  
      Args:
          df (pd.DataFrame): DataFrame containing the calibration data.
          metric (str): Metric to plot on the y-axis.
          label (str): Label for the calibration curve.
          ax (plt.Axes): Matplotlib axes to plot on.
  
      Returns:
          Matplotlib axes with the precision-recall plot.
      """
  
      # Set the style of seaborn for more attractive plots
      sns.set_theme(style="ticks")
  
      if ax is None:
          fig, ax = plt.subplots(figsize=(5, 5))
  
      # Adding the precision-recall curve from dataframe with marker options
      sns.lineplot(x='percentile', y=metric, data=df, ax=ax, label=label, linewidth=1.5)
  
      # Setting labels with increased font size for better readability
      ax.set_xlabel('Percentile', fontsize=12)
      ax.set_ylabel(metric, fontsize=12)
  
      # Set the range of x axis
      ax.set_xlim([-0.02, 1.02])
  
      # Additional customization: removing the top and right spines for a cleaner look
      sns.despine()
  
      return ax




def per_scene_uq_analysis(config) -> None:

    model_checkpoint_paths = config["model_checkpoint_paths"]
    result_dir = config["output_dir"]
    device = "cpu"
    if config["device"] == "gpu":
        device = "cuda"
    processes = None·

    result_dir = Path(result_dir)
    #result_dir.mkdir(parents=True, exist_ok=True)

    model = EnsembleModule(
        checkpoint_paths=[model_checkpoint_paths[0]], #, model_checkpoint_paths[0]],
        monte_carlo_steps=0,
        return_raw_predictions=True
    )   
    #model = EvidentialUnetModel.load_from_checkpoint(model_checkpoint_paths[1])
    model.to(device)
·



def uq_analysis(
        config
    ) -> None:


    model_checkpoint_paths = config["model_checkpoint_paths"]
    result_dir = config["output_dir"]
    device = "cpu"
    if config["device"] == "gpu":
        device = "cuda"
    processes = None 

    result_dir = Path(result_dir)
    #result_dir.mkdir(parents=True, exist_ok=True)

    model = EnsembleModule(
        checkpoint_paths=[model_checkpoint_paths[0]], #, model_checkpoint_paths[0]],
        monte_carlo_steps=0,
        return_raw_predictions=True
    )
    #model = EvidentialUnetModel.load_from_checkpoint(model_checkpoint_paths[1])
    model.to(device)
 
    # for noise_level in [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]:
    dm = dm = CloudDataModule(config)
    dm.prepare_data()
    dm.setup()
    dataset_name = "cloud_data_test_set"
    for noise_level in [0.0, 0.02, 0.06, 0.2, 0.5]: # , 0.02, 0.04]:
 
        dataset = dm.data_test
    
        print(f"Making predictions on {dataset_name}...")
        inputs, y_preds, y_trues, aleatoric_vars, epistemic_vars, combined_vars = make_predictions(
                model=model,
                dataset=dataset,
                batch_size=5,
                device=device,
                epsilon=noise_level,
        )

        print(f"Saving predictions on {dataset_name}...")
        np.save(result_dir / f"{dataset_name}_{noise_level}_inputs.npy", inputs.numpy())
        np.save(result_dir / f"{dataset_name}_{noise_level}_y_preds.npy", y_preds.numpy())
        np.save(result_dir / f"{dataset_name}_{noise_level}_y_trues.npy", y_trues.numpy())
        np.save(result_dir / f"{dataset_name}_{noise_level}_aleatoric_vars.npy", aleatoric_vars.numpy())
        np.save(result_dir / f"{dataset_name}_{noise_level}_epistemic_vars.npy", epistemic_vars.numpy())
            
        print(f"Computing metrics on {dataset_name}...")
        df = convert_to_pandas(
            y_preds=y_preds,
            y_trues=y_trues,
            aleatoric_vars=aleatoric_vars,
            epistemic_vars=epistemic_vars,
            combined_vars=combined_vars,
        )
        df = compute_metrics(df)

        print(f"Saving dataframes for {dataset_name}...")
        df.to_pickle(result_dir / f"{dataset_name}_{noise_level}_metrics.pkl")
            
        print(f"Creating data for precision-recall plot on {dataset_name}...")
        df_cutoff = create_precision_recall_plot(df)
        df_cutoff.to_csv(result_dir / f"{dataset_name}_{noise_level}_precision_recall.csv", index=False)

        ax = plot_precision_recall(df_cutoff)
        plt.savefig(result_dir / f"{dataset_name}_{noise_level}_precision_recall.png")
        plt.clf()

        print(f"Creating data for calibration plot on {dataset_name}...")
        processes = mp.cpu_count() if processes is None else processes
        df_subset = df.iloc[:10000008]
        df_calibration = create_calibration_plot(df_subset, scipy.stats.laplace, processes=processes)
        df_calibration.to_csv(result_dir / f"{dataset_name}_{noise_level}_calibration.csv", index=False)

        ax = plot_calibration(df_calibration)
        plt.savefig(result_dir / f"{dataset_name}_{noise_level}_calibration.png")
        plt.clf()


        print(f"Finished processing dataset `{dataset_name}`!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML config.")
    args = parser.parse_args()

    config = utils.read_yaml(args.yaml)

    #logger.debug("command line arguments: %s", args)
    uq_analysis(config)



