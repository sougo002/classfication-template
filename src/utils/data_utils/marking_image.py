import imghdr
from math import sqrt
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

# parent_dir以下にある画像をディレクトリ構造そのままにoutput_dir以下にn分割して保存する
parent_dir = Path(r'C:\Users\s.nakamura\workspace\projects\shimaseiki\datasets\元画像')
output_dir = Path(r'C:\Users\s.nakamura\workspace\projects\shimaseiki\datasets\lined')
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
        if not imghdr.what(file):
            continue
        img = Image.open(file)
        original_img = np.array(img)
        h, w = original_img.shape[:2]
        h = h // part
        w = w // part
        # n個にわける
        for i in range(part):
            for j in range(part):
                h_s = h*i-offset if h*i - offset >= 0 else 0
                h_e = h*(i+1)+offset if h*(i+1)+offset < h*part else h*part-1
                w_s = w*j-offset if w*j - offset >= 0 else 0
                w_e = w*(j+1)+offset if w*(j+1)+offset < w*part else w*part-1
                draw = ImageDraw.Draw(img)
                draw.line(((w_s, h_e), (w_e, h_e)), fill=(0, 0, 0), width=3)
                draw.line(((w_e, h_s), (w_e, h_e)), fill=(0, 0, 0), width=3)
        img.save(output_dir / relative_path / (file.stem + '_line.png'))


if __name__ == '__main__':
    main()
