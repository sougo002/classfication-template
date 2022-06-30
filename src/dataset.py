from PIL import Image
from torch.utils.data import Dataset

from pathlib import Path
import os


class ImageDataset(Dataset):
    """
    Image Dataset class, inheriting from PyTorch's Dataset class.

    Attributes
    ----------
    file_list : list
        A list of image paths
    transform : object
        An insctance of pre-processing class
    phase : 'train' or 'test'
        Training mode or test mode
    """

    def __init__(self, df, transform, cfg_dataset, root_dir: Path = None):
        file_list = df.path.to_list()
        labels = df.label.to_list()
        self.file_list = file_list
        self.labels = labels
        self.transform = transform
        self.cfg_dataset = cfg_dataset
        self.root_dir = root_dir if root_dir is not None else Path()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path, target = self.file_list[index], self.labels[index]
        img = Image.open(self.root_dir/img_path.replace('\\', os.sep)).convert('RGB')
        img_transformed = self.transform(img)

        return img_transformed, target
