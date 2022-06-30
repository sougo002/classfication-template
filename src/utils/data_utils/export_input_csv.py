import imghdr
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split

# データディレクトリ
dataset_dir = Path(r'C:\Users\s.nakamura\workspace\projects\shimaseiki\datasets')
# プロジェクトルートディレクトリ
root_dir = Path(r'C:\Users\s.nakamura\workspace\projects\shimaseiki')
set_num = 1
data_dirs = ['split_normal',
             'split_anomaly',
             ]
dir_labels = [0, 1]

paths = []
labels = []

for dir, label in zip(data_dirs, dir_labels):
    dir = dataset_dir / dir
    for file in dir.iterdir():
        if imghdr.what(file):
            paths.append(file.relative_to(root_dir))
            labels.append(label)

file_df = pd.DataFrame({'label': labels, 'path': paths})
train_df, test_df = train_test_split(file_df, test_size=0.2, random_state=46, stratify=file_df['label'])
print(train_df)
print(test_df)
csv_dir = Path(root_dir / 'inputs')
csv_dir.mkdir(exist_ok=True, parents=True)
train_df.to_csv(csv_dir / f'set{set_num}_train.csv', index=False)
test_df.to_csv(csv_dir / f'set{set_num}_test.csv', index=False)
