from typing import List
from argparse import Namespace, ArgumentParser
from datetime import datetime
import logging

from functools import partial

import torch
import wandb

import lightning.pytorch as pl
from pytorch_lightning.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger

from mimo.utils import dir_path
from mimo.models.mimo_unet import MimoUnetModel
from mimo.models.evidential_unet import EvidentialUnetModel

from cloud_datamodule import CloudDataModule
from callbacks import OutputMonitor, WandbMetricsDefiner

from torchvision import transforms
from generate_training_dataset import preprocess_global
from cloud_unet_pl import CloudUNetPL
import utils

from pprint import pprint

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def default_callbacks(validation: bool = True) -> List[pl.Callback]:
    callbacks = [
        #OutputMonitor(),
        ModelCheckpoint(dirpath="./", filename="cloud_unet", every_n_epochs=1, save_last=True),
    ]
    if validation:
        callbacks_validation = [
            ModelCheckpoint(
                monitor="val_loss",
                save_top_k=1,
                #filename="epoch-{epoch}-step-{step}-valloss-{val_loss:.8f}-mae-{metric_val/mae_epoch:.8f}",
                auto_insert_metric_name=True,
            ),
            #ModelCheckpoint(dirpath="./", filename="cloud_mimo_unet", every_n_epochs=1, save_last=True),
            WandbMetricsDefiner(),
        ]
        callbacks += callbacks_validation
    return callbacks



def main(config):

    if config["run_tune"]:
        run = wandb.init()
        lr = wandb.config.lr
        bs = wandb.config.batch_size
        epochs = wandb.config.epochs
        insc = wandb.config.input_scaler
        tarsc = wandb.config.target_scaler
        login = wandb.config.log_scale_input
        logtar = wandb.config.log_scale_target
        config["epochs"] = epochs
        config["log_scale_input"] = login
        config["log_scale_target"] = logtar
        config["input_scaler"] = insc
        config["target_scaler"] = tarsc
        config["batch_size"] = bs
        config["learning_rate"] = lr
        config["load_and_scale"] = False

        if config["use_mimo"]:
            config["mimo_num_subnetworks"] = wandb.config.mimo_num_subnetworks
            config["mimo_filter_base_count"] = wandb.config.mimo_filter_base_count
            config["mimo_loss_buffer_size"] = wandb.config.mimo_filter_base_count
            config["mimo_loss_buffer_temperature"] = wandb.config.mimo_loss_buffer_temperature
            config["mimo_loss"] = wandb.config.mimo_loss


    pl.seed_everything(config["rng_seed"])
    dm = CloudDataModule(config)
    dm.prepare_data()
    dm.setup()

 
    if config["use_mimo"]:
        model = MimoUnetModel(
            in_channels=dm.data_train.X.shape[1],
            out_channels=dm.data_train.y.shape[1]*2,
            num_subnetworks=config["mimo_num_subnetworks"],
            filter_base_count=config["mimo_filter_base_count"],
            center_dropout_rate=config["mimo_center_dropout_rate"],
            final_dropout_rate=config["mimo_final_dropout_rate"],
            encoder_dropout_rate = config["mimo_encoder_dropout_rate"],
            core_dropout_rate=config["mimo_core_dropout_rate"],
            decoder_dropout_rate=config["mimo_decoder_dropout_rate"],
            loss_buffer_size=config["mimo_loss_buffer_size"],
            loss_buffer_temperature=config["mimo_loss_buffer_temperature"],
            input_repetition_probability=config["mimo_input_repetition_probability"],
            batch_repetitions=config["mimo_batch_repetitions"],
            loss=config["mimo_loss"], #laplace_nll or gaussian_nll
            weight_decay=config["weight_decay"],
            learning_rate = config["learning_rate"],
            seed = config["rng_seed"]
        )
    else:
        model = CloudUNetPL(config, in_channels=dm.data_train.X.shape[1], out_channels=dm.data_train.y.shape[1])


    wandb_logger = WandbLogger(project="Cloud", log_model=True, save_dir=config["checkpoint_path"])

    trainer = pl.Trainer(
        callbacks=default_callbacks(), 
        accelerator='gpu', 
        devices=1,
        strategy="auto", #ddp", #DDPStrategy(find_unused_parameters=True),
        precision="16-mixed",
        max_epochs=config["epochs"],
        default_root_dir=config["checkpoint_path"],
        log_every_n_steps=1,
        logger=wandb_logger,
    )

    torch.set_float32_matmul_precision('high') # | 'high')
    trainer.started_at = str(datetime.now().isoformat(timespec="seconds"))
    #wandb.init()
    trainer.fit(model, dm)
    wandb_logger.experiment.finish()



def tune_model(config):
 

    wandb_sweep_config = {
            "method": "bayes",
            "name": "sweep",
            "metric": {"goal": "minimize", "name": "val_loss"},
            "parameters": {
                "batch_size": {"values": [16, 32, 64]},
                "lr": {"distribution": "log_uniform", "min": 1e-5 , "max": 0.1},
                "input_scaler": {"values": ["MinMax", "Standard", "NONE"]},
                "target_scaler": {"values": ["MinMax", "Standard", "NONE"]},
                "log_scale_input": {"values": [True, False]},
                "log_scale_target": {"values": [True, False]},
                "epochs": {"value" : 30} },
            "early_terminate": {
                "type": "hyperband",
                "min_iter": 5,
                "eta": 2,
                "strict": True
            }
    }

    print(wandb_sweep_config)

    if config["tune_tiles"]:
        wandb_sweep_config["parameters"]["stride"] = {"values": [int(round(config["stride"] / 2)), config["stride"], int(config["stride"] * 2), int(config["stride"] * 4)]}
        wandb_sweep_config["parameters"]["tile_size"] = {"values": [config["tile_size"], int(config["tile_size"] * 2), int(config["tile_size"] * 3), int(config["tile_size"] * 4)]}


    if config["use_mimo"]:
        wandb_sweep_config["parameters"]["mimo_num_subnetworks"] = {"values": [2,3,5,7]}
        wandb_sweep_config["parameters"]["mimo_filter_base_count"] = {"value": 64}
        wandb_sweep_config["parameters"]["mimo_loss_buffer_size"] = {"values": [0,2,5,10,20]}
        wandb_sweep_config["parameters"]["mimo_loss_buffer_temperature"] = {"values": [0.1,0.2,0.5,1.0]}
        wandb_sweep_config["parameters"]["mimo_loss"] = {"values": ["laplace_nll", "gaussian_nll"]}


    pprint(wandb_sweep_config)

    sweep_id = wandb.sweep(sweep=wandb_sweep_config, project="Cloud")
    eval_func = partial(main, config)
    wandb.agent(sweep_id=sweep_id, function=eval_func)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()

    config = utils.read_yaml(args.yaml)

    logger.debug("command line arguments: %s", args)
    if not config["run_tune"]:
        main(config)
    else:
        tune_model(config)

