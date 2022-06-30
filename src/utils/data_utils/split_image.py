import imghdr
from math import sqrt
from pathlib import Path
import numpy as np
from PIL import Image

# parent_dir以下にある画像をディレクトリ構造そのままにoutput_dir以下にn分割して保存する
parent_dir = Path(r'C:\Users\s.nakamura\workspace\projects\shimaseiki\datasets')
output_dir = Path(r'C:\Users\s.nakamura\workspace\projects\shimaseiki\datasets\split')
output_dir.mkdir(parents=True, exist_ok=True)
resize_key = 'lanczos'
RESIZE_FLAG = Image.LANCZOS
# 分割後画像サイズ
SIZE = 512
# 平方数
n_split = 16
# 余分に取るサイズ
offset = 0
part = int(sqrt(n_split))


def main():
    for file in parent_dir.glob('**/*'):
        if file.is_dir():
            if file.name == output_dir.name:
                break
            continue
        relative_path = file.parent.relative_to(parent_dir)
        (output_dir / relative_path).mkdir(parents=True, exist_ok=True)
        index = 1
        if not imghdr.what(file):
            continue
        original_img = np.array(Image.open(file))
        h, w = original_img.shape[:2]
        h = h // part
        w = w // part
        # n個にわける
        for i in range(part):
            for j in range(part):
                h_s, h_e = 0, 0
                w_s, w_e = 0, 0
                h_s = h*i-offset if h*i - offset >= 0 else 0
                h_e = h*(i+1)+offset if h*(i+1)+offset < h*part else h*part-1
                w_s = w*j-offset if w*j - offset >= 0 else 0
                w_e = w*(j+1)+offset if w*(j+1)+offset < w*part else w*part-1
                img = Image.fromarray(original_img[h_s:h_e, w_s:w_e])
                img = img.resize((SIZE, SIZE), RESIZE_FLAG)
                img.save(output_dir / relative_path / (file.stem + f'_part{index}.png'))
                index += 1


if __name__ == '__main__':
    main()
