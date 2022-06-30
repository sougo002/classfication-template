from sklearn.model_selection import train_test_split
from pathlib import Path
import time
import shutil
import imghdr
import numpy as np


np.random.seed(46)


root_dir = Path('./')
print(f'root path: {root_dir.resolve()}')

dataset_dir = Path(r'C:\Users\s.nakamura\workspace\projects\toppan-g\toppan-printing-gunma-plant\images\再検証\混ぜ混ぜデータ\upper_shot1')
print(f'dataset path: {dataset_dir.resolve()}')

train_dir = dataset_dir / 'train'
train_dir.mkdir(exist_ok=True, parents=True)
test_dir = dataset_dir / 'test'
test_dir.mkdir(exist_ok=True, parents=True)

train_list, test_list = train_test_split(list(root_dir.iterdir()),
                                         test_size=0.5,
                                         random_state=46)

# ディレクトリを再帰的に探索、trainとtestに分けて保存
for file in train_list:
    if file.is_file() and imghdr.what(file):
        shutil.copy2(file.resolve(), train_dir / file.name)

for file in test_list:
    if file.is_file() and imghdr.what(file):
        shutil.copy2(file.resolve(), test_dir / file.name)

time.sleep(3)
