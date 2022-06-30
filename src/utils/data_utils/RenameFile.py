# 構造そのままでindex_元画像名にリネームする
# 元画像パスと現画像パスが対応するcsvを作る
from pathlib import Path
import imghdr
import shutil
import pandas as pd

# 画像を探すディレクトリ(再帰的にすべて)
from_dir = Path(r'C:\Users\s.nakamura\workspace\projects\suzuki\poc_insertmold\classification\datasets\PoC検証画像(整理済み)20220222')

# リネームして保存するディレクトリ
to_dir = Path(r'C:\Users\s.nakamura\workspace\projects\suzuki\poc_insertmold\classification\datasets\indexed')
to_dir.mkdir(parents=True, exist_ok=True)

# 除外ディレクトリ
except_dir = ['検出ポイント']


def dfs(parent_dir, index, file_list):
    if parent_dir.name in except_dir:
        return index, file_list
    for file in parent_dir.iterdir():
        if file.is_dir():
            index, file_list = dfs(file, index, file_list)
            continue

        relative_path = parent_dir.relative_to(from_dir)
        save_dir = to_dir / relative_path
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        if imghdr.what(file):
            new_path = save_dir / (f'{str(index)}_{file.name}')
            shutil.copy2(file, new_path)
            # csv追加
            add_list = pd.Series({'new_path': new_path.resolve(), 'old_path': file.resolve()})
            file_list.loc[index] = add_list
            index += 1
    return index, file_list


index = 0
file_list = pd.DataFrame(columns=['new_path', 'old_path'])
index, file_list = dfs(from_dir, index, file_list)
print(f'{index} index images copied')
file_list.to_csv('./file_list.csv')
