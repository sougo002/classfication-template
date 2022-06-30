import torch
from torch import nn
from torchvision import transforms

from models.architecture import EfficientNet, EfficientNet_AdaCos


def get_model(cfg_model):
    model = None
    if cfg_model.name.lower() == 'efficientnet':
        model = EfficientNet(cfg_model.size, len(cfg_model.classes), cfg_model.pretrained)
    if cfg_model.name.lower() == 'efficientnet_adacos':
        model = EfficientNet_AdaCos(cfg_model)
    return model


def get_loss(name: str = 'BCE'):
    loss = None
    if name == 'BCE':
        loss = nn.CrossEntropyLoss()

    if loss is None:
        raise ValueError(f'no loss func named:{name}.')
    return loss


def get_optimizer(cfg_optimizer):
    optim = None
    name = cfg_optimizer.name.lower()
    if name == 'adam':
        optim = torch.optim.Adam
    elif name == 'adamw':
        optim = torch.optim.AdamW
    else:
        optim = torch.optim.SGD

    if optim is None:
        raise ValueError(f'no optimizer named:{name}.')

    return optim


def get_lr_scheduler(cfg_lr_scheduler):
    scheduler = None
    name = cfg_lr_scheduler.name.lower()
    if name == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR
    if scheduler is None:
        raise ValueError(f'no lr_scheduler named:{name}.')

    return scheduler


def get_transform(cfg_augmentation):
    aug_list = []
    for name in cfg_augmentation.list:
        if 'random_crop' == name:
            aug_list.append(transforms.RandomCrop(**cfg_augmentation.random_crop.params))
        if 'sharpness' == name:
            aug_list.append(transforms.RandomAdjustSharpness(**cfg_augmentation.sharpness.params))
        if 'color_jitter' == name:
            aug_list.append(transforms.ColorJitter(**cfg_augmentation.color_jitter.params))
        if 'affine' == name:
            aug_list.append(transforms.RandomAffine(**cfg_augmentation.affine.params))
        if 'horizontal_flip' == name:
            aug_list.append(transforms.RandomHorizontalFlip(0.5))
        if 'vertical_flip' == name:
            aug_list.append(transforms.RandomVerticalFlip(0.5))

    aug_list.append(transforms.Resize((cfg_augmentation.size, cfg_augmentation.size)))
    aug_list.append(transforms.ToTensor())

    return transforms.Compose(aug_list)
