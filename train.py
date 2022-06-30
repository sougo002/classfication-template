# 外部ライブラリ
import torch
import torchvision
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

# インナーコード
from src.lightning_module import CustomLightningModule
from src.data_module import CustomDataModule
from src.utils.time_count import Timer
from src.utils.custom_logging import CustomLogger

# 標準ライブラリ
from pathlib import Path


logger = CustomLogger(__name__).get_logger()

# Version check of Pytorch
logger.info(f'PyTorch Version:{torch.__version__}')
logger.info(f'Torchvision Version:{torchvision.__version__}')

# Constants
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
logger.debug(f'The device used is:{device}')

if device == 'cuda:0':
    gpu_name = torch.cuda.get_device_name()
    logger.info(gpu_name)


# train
@hydra.main(config_path='config')
def train(cfg: DictConfig):
    config_name = HydraConfig.get().job.config_name
    seed_everything(cfg.General.seed)
    is_debug = cfg.General.debug
    fold = cfg.Data.dataset.fold
    timer = Timer()
    timer.record(name='start')
    logger.info('-------------in train------------')
    logger.info(f'config - \n{OmegaConf.to_yaml(cfg)}')

    root_dir = Path(get_original_cwd())
    experiment_dir = Path('./')
    logger.info(f'root dir : {root_dir}')
    logger.info(f'experiment dir : {experiment_dir.resolve()}')
    logger.info(f'config : {config_name}')

    trainer_loggers = [CSVLogger(save_dir=str(experiment_dir), name=f'fold_{fold}'),
                       TensorBoardLogger(save_dir=str(experiment_dir))
                       ]

    checkpoint_callback = ModelCheckpoint(dirpath=str(root_dir/f'outputs/checkpoints/{config_name}'),
                                          filename='fold_'+str(fold)+'_best{val_acc:.2f}',
                                          monitor='val_acc',
                                          save_last=True,
                                          save_top_k=1,
                                          save_weights_only=True,
                                          mode='max')

    trainer = Trainer(
        max_epochs=2 if is_debug else cfg.General.epoch,
        gpus=cfg.General.gpus,
        # accumulate_grad_batches=cfg.General.grad_acc,
        precision=16 if cfg.General.fp16 else 32,
        # amp_level=cfg.General.amp_level,
        # amp_backend='native',
        # deterministic=True,
        # auto_select_gpus=False,
        # benchmark=False,
        default_root_dir=str(experiment_dir),
        limit_train_batches=0.02 if is_debug else 1.0,
        limit_val_batches=0.05 if is_debug else 1.0,
        callbacks=[checkpoint_callback],
        logger=trainer_loggers,
    )

    model = CustomLightningModule(cfg)
    datamodule = CustomDataModule(cfg)

    trainer.fit(model=model, datamodule=datamodule)

    timer.record(name='end')
    logger.info(f'elapsed time : {str(timer.get_elapsed_time())}')


if __name__ == '__main__':
    train()
