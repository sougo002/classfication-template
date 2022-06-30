import imghdr
from pathlib import Path
import shutil
from PIL import Image

# 画像を探すディレクトリ(再帰的にすべて)
from_dir = Path(rf'C:\Users\s.nakamura\workspace\projects\suzuki\poc_insertmold\classification\datasets\PoC検証画像(整理済み)20220222\過検知900枚\画像')

# 拡張子を変えて保存するディレクトリ
to_dir = Path(rf'C:\Users\s.nakamura\workspace\projects\suzuki\poc_insertmold\classification\datasets\PoC検証画像(整理済み)20220222\過検知900枚\過検知画像')
to_dir.mkdir(parents=True, exist_ok=True)

SIZE = 1280


def recursive_search(parent_dir):
    for file in parent_dir.iterdir():
        if file.is_dir():
            recursive_search(file)

        relative_path = parent_dir.relative_to(from_dir)
        save_dir = to_dir / relative_path
        save_dir.mkdir(parents=True, exist_ok=True)

        if file.is_dir() or (not imghdr.what(file)):
            continue
        img = Image.open(file).convert('RGB')
        #img = img.resize((SIZE, SIZE), Image.LANCZOS)
        img.save(save_dir / (file.stem + '.jpg'), quality=100)


recursive_search(from_dir)
