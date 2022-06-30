import pytorch_lightning as pl
from torch.utils.data import DataLoader
import pandas as pd
from hydra.utils import get_original_cwd
from sklearn.model_selection import StratifiedKFold, train_test_split

from dataset import ImageDataset
from factory import get_transform

from utils.custom_logging import CustomLogger
from pathlib import Path


logger = CustomLogger(__name__).get_logger()


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super(CustomDataModule, self).__init__()

        # Config読み込み
        self.cfg_general = cfg.General
        self.cfg_dataset = cfg.Data.dataset
        self.cfg_dataloader = cfg.Data.dataloader
        self.cfg_augmentation = cfg.Augmentation
        self.root_dir = Path(get_original_cwd())
        self.train_df = None
        self.valid_df = None
        self.test_df = None

    def get_train_valid_df(self):
        # 学習，バリデーションdfを返す(stratified k-fold)
        df = pd.read_csv(self.root_dir / self.cfg_dataset.train_csv)
        logger.debug(f'train csv path:{(self.root_dir / self.cfg_dataset.train_csv).resolve()}')
        if self.cfg_dataset.fold_k != 1:
            skf = StratifiedKFold(n_splits=self.cfg_dataset.fold_k, shuffle=True, random_state=self.cfg_general.seed)
            for fold, (_, valid_index) in enumerate(skf.split(df, df['label'])):
                df.loc[valid_index, 'fold'] = fold+1
            df['fold'] = df['fold'].astype('int')
            train_df = df[df['fold'] != int(self.cfg_dataset.fold)].reset_index(drop=True)
            valid_df = df[df['fold'] == int(self.cfg_dataset.fold)].reset_index(drop=True)
            return train_df, valid_df
        else:
            file_df = pd.read_csv(self.root_dir / self.cfg_dataset.train_csv)
            return train_test_split(file_df, test_size=self.cfg_dataset.valid_size, random_state=self.cfg_general.seed, stratify=file_df['label'])

    def get_test_df(self):
        # valid == test
        if self.cfg_general.no_valid:
            logger.warning(f'No validation is NOT RECOMMENDED. over-fitting to test data may cause some issue during opration.')
            return self.valid_df
        logger.debug(f'test csv path:{(self.root_dir / self.cfg_dataset.test_csv).resolve()}')
        return pd.read_csv(self.root_dir / self.cfg_dataset.test_csv)

    def setup(self, stage):
        logger.info(f'in setup. stage : {stage}')
        self.train_df, self.valid_df = self.get_train_valid_df()
        self.test_df = self.get_test_df()

    def get_dataframe(self, phase):
        assert phase in {"train", "valid", "test"}
        if phase == "train":
            return self.train_df
        elif phase == "valid":
            return self.valid_df
        elif phase == "test":
            return self.test_df

    def get_ds(self, phase):
        assert phase in {"train", "valid", "test"}
        transform = get_transform(cfg_augmentation=self.cfg_augmentation[phase])
        ds = ImageDataset(
            df=self.get_dataframe(phase=phase),
            transform=transform,
            cfg_dataset=self.cfg_dataset,
            root_dir=self.root_dir
        )
        return ds

    def get_loader(self, phase):
        dataset = self.get_ds(phase=phase)
        return DataLoader(
            dataset,
            batch_size=self.cfg_dataloader.batch_size,
            shuffle=True if phase == "train" else False,
            num_workers=self.cfg_dataloader.num_workers,
            drop_last=True if phase == "train" else False,
        )

    # Trainer.fit() 時に呼び出される
    def train_dataloader(self):
        return self.get_loader(phase="train")

    # Trainer.fit() 時に呼び出される
    def val_dataloader(self):
        return self.get_loader(phase="valid")