# 外部ライブラリ
import torch
import torchvision
from pytorch_lightning.utilities.seed import seed_everything
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from PIL import Image
import numpy as np

# インナーコード
from src.utils.time_count import Timer
from src.utils.custom_logging import CustomLogger
from src.utils.utils import draw_hist, export_wrong_images, roc_plot, draw_confusion_matrix

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


# test
@hydra.main(config_path='config', version_base=None)
def evaluate(cfg: DictConfig):
    config_name = HydraConfig.get().job.config_name
    seed_everything(cfg.General.seed)
    fold = cfg.Data.dataset.fold
    timer = Timer()
    timer.record(name='start')
    logger.info('-------------in evaluate------------')
    logger.info(f'config - \n{OmegaConf.to_yaml(cfg)}')

    root_dir = Path(get_original_cwd())
    experiment_dir = Path('./')
    logger.info(f'root dir : {root_dir}')
    logger.info(f'experiment dir : {experiment_dir.resolve()}')
    logger.info(f'config : {config_name}')

    result = pd.read_csv(root_dir / cfg.Evaluate.result_file)

    save_dir = root_dir / 'outputs' / config_name / fold

    labels = np.array(result['label'])
    preds = np.array(result['prediction'])

    # acc
    count = 0
    correct_num = 0
    for label, pred in zip(labels, preds):
        count += 1
        if label == pred:
            correct_num += 1
    logger.info(f'normal acc: {correct_num / count}')

    # hist
    scores = np.array((torch.tensor(result['anomaly'].to_list()).sigmoid() * 100).tolist())
    draw_hist(labels, scores, save_dir)

    # roc, auroc
    fpr, tpr, threshold = roc_plot(labels, scores, save_dir)
    logger.debug(f'fpr:{fpr}, tpr:{tpr}\n threshold:{threshold}')

    # 精度最良のconfusion matrix
    cm = draw_confusion_matrix(labels, preds, save_dir=save_dir, name='acc_confusion matrix')
    # 見逃しゼロのconfusion matrix
    cm = draw_confusion_matrix(labels, scores, threshold[-2], save_dir=save_dir)
    tn = cm[0][0]
    fp = cm[1][0]
    od_rate = fp / (tn + fp)
    logger.info(f'見逃し0の過検知率:{od_rate}')
    # 間違い画像出力
    export_wrong_images(result=result, root_dir=root_dir, output_dir=save_dir)

    timer.record(name='end')
    logger.info(f'elapsed time : {str(timer.get_elapsed_time())}')


if __name__ == '__main__':
    evaluate()
