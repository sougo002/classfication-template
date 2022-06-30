import sys
sys.path.append(r'C:\Users\s.nakamura\workspace\projects\template')

import csv
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from pathlib import Path
from utils.utils import seed_everything
import shutil
import imghdr
import numpy as np


seed_everything(torch=False)

fold = 3

csv_file = pd.read_csv(r'C:\Users\s.nakamura\workspace\dataset\DAGM2007\inputs\set1_dataset.csv')

print(csv_file)

kf = KFold(n_splits=fold, shuffle=True, random_state=46)

labels = csv_file['label']
path_list = csv_file['path']
print(labels, path_list)
exit()
