from typing import Optional
import lightning.pytorch as pl
import torch
import utils
from torchvision import transforms
from generate_training_dataset import preprocess_global
from DataLoader import Cloud_Dataset, ToTensor, MultiAngleRandomHorizontalFlip

def load_and_preprocess_train(config):
    if config["load_and_scale"]:
        train_test_val_dataset = utils.load_train_test_val_dataset(config['out_dir'])

    else:
        input_tiled, target_tiled, nx, ny = preprocess_global(config)
        # save results to netCDF
        utils.save2dataset(input_tiled, target_tiled, nx, ny, config)
        print(f"rad tiled {input_tiled.shape}")
        input_tiles_scaled, target_tiles_scaled = utils.scale_training_data(input_tiled, target_tiled, config)
        # split test/train/val
        train_test_val_dataset = utils.preprocess_training_data(input_tiles_scaled.transpose(0,3,1,2), target_tiles_scaled.transpose(0,3,1,2), config)
    print(f"train shape {train_test_val_dataset[0].shape}")
    return train_test_val_dataset


class CloudDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.config = config
        self.batch_size = self.config["batch_size"]


    def setup(
        self,
        stage: Optional[str] = None,
    ) -> None:

        # load and pre-process dataset
        x_train, x_test, x_val, y_train, y_test, y_val = load_and_preprocess_train(self.config)
        print(f"shape before training: X {x_train.shape}, y {y_train.shape}")

        print("DATA STATS", x_train.min(), x_train.mean(), x_train.std(), x_train.max(), y_train.min(), y_train.mean(), y_train.std(), y_train.max(), x_train.shape)

        # set up transformations
        transformations = [transforms.ToTensor()]


        # transformations for test/val (no data augmentation)
        im_transform = transforms.Compose(transformations)
        if "transform_fliph_prob" in self.config.keys():
            transformations.append(
                MultiAngleRandomHorizontalFlip(p=self.config["transform_fliph_prob"])
                )  
        if "transform_flipv_prob" in self.config.keys():
            transformations.append(
                transforms.RandomVerticalFlip(p=self.config["transform_flipv_prob"]),
                )
        # transformations for training (optional data augmentation)
        train_transform = transforms.Compose(transformations)

        self.data_train = Cloud_Dataset(
            x_train, y_train,
            transform_image=train_transform)

        self.data_valid = Cloud_Dataset(
            x_val, y_val,
            transform_image = im_transform)

        self.data_test = Cloud_Dataset(
            x_test, y_test,
            transform_image = im_transform)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.data_train,
            batch_size=self.config["batch_size"],
            num_workers=32, 
            drop_last= True,
            shuffle=True,
            persistent_workers=True
        )
    
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.data_valid,
            batch_size=self.config["batch_size"],
            num_workers=32,
            shuffle=False,
            persistent_workers=True
        )
    
    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.data_test,
            batch_size=self.config["batch_size"],
            num_workers=32,
            shuffle=False,
            persistent_workers=True
        )
    
