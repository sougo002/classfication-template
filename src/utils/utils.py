import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision.transforms.functional as TF
# Grad-CAM
from gradcam import GradCAM, GradCAMpp
from gradcam.utils import visualize_cam
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from torchmetrics import ConfusionMatrix
from torchvision.utils import make_grid, save_image


def seed_everything(seed=46, torch=False):
    # Fix random seeds
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    if torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def show_grid(images, nrow=5, save_dir=None):
    grid_image = make_grid(images, nrow=nrow, padding=5)
    if save_dir:
        save_image(grid_image, fp=save_dir, nrow=5, padding=2)


def create_heatmap(net, target_layer, path, device, transform, save_dir=None,
                   is_pp=True, add_normal=True, class_idx=None):
    """
    Create heatmap images in grid view.

    Parameters
    ----------
    save_dir : str or None
        Specify a directory and file name if you want to save the heatmap image.
    is_pp : bool
        use GradCam++?
    add_normal : bool
        add normal images next to heatmap
    class_idx : int or None
        specify the class index in heatmap
    """
    # Grad-CAM
    net.eval()

    res_img = []
    img = Image.open(path).convert('RGB')
    torch_img = transform(img).to(device)[None]

    if is_pp:
        gradcam_pp = GradCAMpp(net, target_layer)
        mask, _ = gradcam_pp(torch_img, class_idx=class_idx)
        _, result = visualize_cam(mask, torch_img)
    else:
        gradcam = GradCAM(net, target_layer)
        mask, _ = gradcam(torch_img, class_idx=class_idx)
        _, result = visualize_cam(mask, torch_img)

    if add_normal:
        res_img.extend([torch_img.to('cpu')[0], result])
    else:
        res_img = result
    show_grid(images=res_img, save_dir=save_dir)


# 間違っている画像を保存する
def export_wrong_images(result, root_dir, output_dir=Path('./'), class_len=2, max_count=25):
    all_images = []
    if not type(result) == pd.core.frame.DataFrame:
        result = pd.read_csv(result)
    for label in range(class_len):
        label_dir = output_dir / f'class{label}'
        label_dir.mkdir(parents=True, exist_ok=True)
        images = []
        index = 1
        # 0:index, 1:label, 2:pred class, 3:ImagePath, 4~:scores, -1: heatmap_path
        for row in result.itertuples():
            if row[1] != row[2] and row[1] == label:
                file = Path(row[3].replace('\\', os.sep))
                # 元ラベルごとの間違い画像をdirに出力
                shutil.copy(root_dir / file, label_dir / f'{index}_wrong_prediction{row[2]}.png')
                image = TF.to_tensor((Image.open(root_dir / file).convert("RGB")))
                if len(images) < max_count:
                    images.append(image)
                if len(all_images) < max_count:
                    all_images.append(image)
                index += 1
        if len(images) != 0:
            show_grid(images, save_dir=output_dir / f'wrong_label{label}.png')
        else:
            print(f'{label} is empty')
    if len(all_images) != 0:
        show_grid(all_images, save_dir=output_dir / f'wrong_all.png')


def export_heatmap_images(result, model, target_layer, output_dir=Path('./'), device=None, transform=None, is_add_normal=True):
    heatmap_dir = output_dir / 'heatmaps'
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    if not type(result) == pd.core.frame.DataFrame:
        result = pd.read_csv(result)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 0:index, 1:label, 2:pred class, 3:ImagePath, 4~:scores, -1: heatmap_path
    for row in result.itertuples():
        file = Path(row[3])
        create_heatmap(model,
                       target_layer=target_layer,
                       path=file,
                       device=device,
                       transform=transform,
                       save_dir=heatmap_dir / f'label{row[1]}_{file.stem}_heatmap.png',
                       add_normal=is_add_normal)


def draw_hist(labels: np.array, scores, save_dir=None, label_names=['normal', 'anomaly'], stacked=False, name=None):
    colors = ['b', 'r', 'y', 'm', 'g', 'c']

    fig = plt.figure()
    plt.title('Score Histogram')
    score_list = []
    # 色をそれぞれ変更するためにラベルごとにスコアを分ける
    for label in np.unique(labels):
        score = scores[labels == label]
        score_list.append(score)
    plt.hist(score_list,
             bins=100,
             range=(scores.min() - 0.05, scores.max() + 0.05),
             alpha=0.5,
             histtype='barstacked' if stacked else 'stepfilled',
             color=colors[:len(score_list)],
             label=label_names)
    plt.xlabel('Anomaly Score')
    plt.ylabel('Count')
    plt.legend()
    if save_dir:
        if name is None:
            name = 'histgram_stack.png' if stacked else 'histgram.png'
        fig.savefig((save_dir/name).resolve())
    plt.close()


def draw_confusion_matrix(y_true, y_pred, threshold=None, num_classes=2, save_dir=None, name=None):
    cm = None
    if threshold is not None:
        confusion_matrix_Maker = ConfusionMatrix(num_classes=num_classes, threshold=threshold)
        # sk learnと形を揃えるため転置
        cm = torch.t(confusion_matrix_Maker(torch.tensor(y_pred), torch.tensor(y_true))).tolist()
    else:
        cm = confusion_matrix(y_pred, y_true)
    sns.heatmap(cm, square=True, cbar=True, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('true')
    plt.ylabel('pred')
    if save_dir:
        if name is None:
            plt.savefig(save_dir / 'confusion_matrix.png')
        else:
            plt.savefig(save_dir / (name + '.png'))
    plt.close()

    return cm


def roc_plot(y_true, y_score, save_dir=None):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fig = plt.figure()
    auc_score = round(roc_auc_score(y_true, y_score), 3)
    plt.title(f'ROC{auc_score}')
    plt.plot(fpr, tpr, marker='o')
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.grid()
    print(f'AUC SCORE: {auc_score}')
    if save_dir:
        fig.savefig((save_dir/'roc.png').resolve())
    plt.close()

    return fpr, tpr, thresholds
