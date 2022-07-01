# 外部ライブラリ
import torch
import torchvision
from pytorch_lightning.utilities.seed import seed_everything
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import numpy as np

# インナーコード
from src.utils.time_count import Timer
from src.utils.custom_logging import CustomLogger
from src.utils.utils import draw_hist, export_wrong_images, draw_confusion_matrix
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
@hydra.main(config_path='config')
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

    save_dir = root_dir / 'outputs' / config_name

    # acc
    count = 0
    correct_num = 0
    for row in result.itertuples():
        count += 1
        if row[1] == row[2]:
            correct_num += 1
    logger.info(f'acc: {correct_num / count}')

    # hist
    labels = np.array(result['label'])
    preds = np.array(result['prediction'])
    scores_list = []
    for anomaly_class in cfg.Model.anomaly_classes:
        ano_scores = np.array((torch.tensor(result[anomaly_class].to_list()).sigmoid() * 100).tolist())
        draw_hist(labels, ano_scores, save_dir, label_names=list(cfg.Model.classes), name=f'{anomaly_class}_hist.png')
        scores_list.append(ano_scores)
    scores = [-1] * len(scores_list[0])  # 異常クラスの最大スコア,最小値で初期化
    for i in range(len(scores_list[0])):
        for k in range(len(scores_list)):
            scores[i] = max(scores_list[k][i], scores[i])

    # TODO: 複数ラベルをまとめて2値分類にする場合の処理
    # # 〇〇のラベルを異常に
    # labels[labels==2] = 1
    # # 〇〇のラベルを異常に
    # labels[labels==3] = 0
    # roc, auc
    #fpr, tpr, threshold = roc_plot(labels, scores, save_dir)
    #logger.debug(f'fpr:{fpr}, tpr:{tpr}\n threshold:{threshold}')
    # 見逃しゼロのconfusion matrix
    # cm = draw_confusion_matrix(labels, scores, threshold[-2], save_dir=save_dir)

    # 精度最良のconfusion matrix
    cm = draw_confusion_matrix(labels, preds, save_dir=save_dir, name='max_confidence_confusion matrix')
    logger.info(cm)
    # 間違い画像出力
    export_wrong_images(result=result, root_dir=root_dir, output_dir=root_dir / 'outputs' / config_name)


    timer.record(name='end')
    logger.info(f'elapsed time : {str(timer.get_elapsed_time())}')

if __name__ == '__main__':
    evaluate()