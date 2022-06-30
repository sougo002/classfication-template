from pathlib import Path
import imghdr
from PIL import Image
import pandas as pd
import shutil

# 画像を探すディレクトリ(再帰的にすべて)
from_dir = Path(r'C:\Users\s.nakamura\workspace\projects\kirishima-shuzou\datasets')

# クロップして保存するディレクトリ
to_dir = Path(r'C:\Users\s.nakamura\workspace\projects\kirishima-shuzou\datasets\cropped')
to_dir.mkdir(parents=True, exist_ok=True)

except_dir = ['', 'cropped']
l, t, r, b = 0, 0, 640, 256
ext = '.bmp'

for file in from_dir.glob('**/*'+ext):
    if file.parent.parent.name in except_dir:
        continue
    save_dir = (to_dir / file.relative_to(from_dir)).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    if imghdr.what(file):
        img = Image.open(file).convert('RGB')
        # クロップ、保存
        img = img.crop((l, t, r, b))
        img.save(save_dir / (file.stem + '.png'))
