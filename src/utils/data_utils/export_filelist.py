from pathlib import Path
from typing import List


def export_filelist(file_list: List[str], label_list: List[str], save_file: Path = Path('./file_list.txt')) -> None:
    save_file.parent.mkdir(parents=True, exist_ok=True)
    if not save_file.exists():
        save_file.touch()
    if len(file_list) != len(label_list):
        raise Exception('ファイルリストとラベルリストの長さは同じである必要があります')
    with save_file.open(mode='w') as f:
        for i in range(len(file_list)):
            f.write(str(label_list[i]) + ',')
            f.write(Path(file_list[i]).resolve())
            f.write('\n')


def import_filelist(import_file: Path = Path('./file_list.txt')):
    file_list, label_list = [], []
    with import_file.open() as f:
        lines = f.readlines()
        for line in lines:
            label, file = line.split(',')
            label_list.append(int(label))
            file_list.append(file.rstrip())
    return (label_list, file_list)
