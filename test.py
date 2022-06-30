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

# インナーコード
from src.lightning_module import CustomLightningModule
from src.data_module import CustomDataModule
from src.utils.time_count import Timer
from src.utils.custom_logging import CustomLogger
from src.utils.utils import create_heatmap
from src.factory import get_transform

# 標準ライブラリ
from pathlib import Path
import os

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
def test(cfg: DictConfig):
    config_name = HydraConfig.get().job.config_name
    seed_everything(cfg.General.seed)
    is_debug = cfg.General.debug
    fold = cfg.Data.dataset.fold
    timer = Timer()
    timer.record(name='start')
    logger.info('-------------in test------------')
    logger.info(f'config - \n{OmegaConf.to_yaml(cfg)}')

    root_dir = Path(get_original_cwd())
    experiment_dir = Path('./')
    logger.info(f'root dir : {root_dir}')
    logger.info(f'experiment dir : {experiment_dir.resolve()}')
    logger.info(f'config : {config_name}')

    model = CustomLightningModule(cfg)

    # model 検証
    classes = cfg.Model.classes
    model_file = root_dir / Path(f'{cfg.Test.model_file}')
    model = model.load_from_checkpoint(model_file, map_location=device, cfg=cfg)

    label_list = []
    pred_list = []
    path_list = []
    score_list = []
    indivisual_score_list = {}
    # 辞書初期化
    for i in range(len(classes)):
        indivisual_score_list[i] = []

    # スコア算出
    i = 0
    with torch.no_grad():
        model = model.to(device)
        model.eval()
        test_csv = pd.read_csv(root_dir / cfg.Data.dataset.test_csv)
        test_data, test_label = test_csv.path.to_list(), test_csv.label.to_list()
        transform = get_transform(cfg_augmentation=cfg.Augmentation['test'])
        timer.record(name='start_infer')
        for test_path, label in zip(test_data, test_label):
            i = i + 1
            label_list.append(label)
            path_list.append(test_path)
            # データ読込
            img = Image.open(root_dir / Path(test_path.replace('\\', os.sep))).convert('RGB')
            img_transformed = transform(img)
            img_transformed = torch.unsqueeze(img_transformed, 0)
            pred = model(img_transformed.to(device)).cpu()
            pred_list.append(pred.argmax(dim=1).detach().numpy()[0])
            score_list.append(pred[0, :len(classes)].detach().numpy())
            for i in range(len(classes)):
                indivisual_score_list[i].append(pred[0, i].detach().numpy())
        timer.record(name='end_infer')
    logger.info(f'infer time avg. {timer.get_time_between("start_infer", "end_infer") / i}')

    result = pd.DataFrame({'label': label_list, 'prediction': pred_list, 'Image path': path_list})
    for i in range(len(classes)):
        result[classes[i]] = indivisual_score_list[i]
    # heatmap class index
    class_idx = None

    export_path = root_dir / 'outputs' / config_name / 'heatmaps' / f'fold{str(fold)}'
    for folder in classes:
        empty_folder = export_path / folder
        empty_folder.mkdir(parents=True, exist_ok=True)

    if cfg.Test.export_heatmap:
        save_path_list = []
        for row in result.itertuples():
            # 0:index, 1:label, 2:pred class, 4:ImagePath, 5~:scores
            original_img_path = root_dir / Path(row[3].replace('\\', os.sep))
            save_path = export_path / classes[row[1]] / original_img_path.name
            save_path_list.append(save_path)
            # ヒートマップ出力 class_idxでどのクラスのヒートマップを出力するかを設定 Noneなら一番スコアが高いクラスになるはず
            create_heatmap(model.get_net().model,
                           target_layer=model.get_net().model.features[-2][0],
                           path=original_img_path,
                           device=device,
                           transform=transform,
                           save_dir=save_path,
                           add_normal=cfg.Test.add_normal,
                           class_idx=class_idx)

        result['Prediction Image Path'] = save_path_list
    result_dir = root_dir / 'outputs/results'
    result_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(result_dir / f'{config_name}.csv', index=False)

    timer.record(name='end')
    logger.info(f'elapsed time : {str(timer.get_elapsed_time())}')


if __name__ == '__main__':
    test()
