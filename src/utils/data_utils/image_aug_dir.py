import cv2
import imghdr
import numpy as np
from aisia_ad_ai_sdk.preprocess import PreprocessUtil
from pathlib import Path


def main(input_base_dir, input_file, output_dir, brightness):
    if input_file is None:
        return
    if input_file.is_dir():
        for file in input_file.iterdir():
            main(input_base_dir, file, output_dir, brightness)
        return
    if imghdr.what(input_file) is not None:
        image_np = PreprocessUtil.read_image_as_np(input_file, mode='color')
        image_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(image_gray[np.nonzero(image_gray)])
        adjust_value = int(brightness - mean_brightness)
        magnificated_img = np.copy(image_np)
    # uint8にキャストするときに255を超えるとオーバーフローしてしまうのでクリップ
        magnificated_img = np.clip((magnificated_img + adjust_value), 0, 255)
        magnificated_img = np.uint8(magnificated_img)
        relative_path = input_file.relative_to(input_base_dir)
        output_target_file = output_dir / relative_path
        output_target_dir = output_target_file.parent
        output_target_file = output_target_dir / f'{input_file.stem}_{brightness}{input_file.suffix}'
        if not output_target_dir.exists():
            output_target_dir.mkdir(parents=True)
        cv2.imwrite(str(output_target_file), magnificated_img)


if __name__ == '__main__':
    input_image_dir = Path('C:/Users/s.nakamura/workspace/projects/aicello/test_data')
    # input_image_dir = Path('C:/work/Projects/action_plan/images/aicello/poc_test_images')
    output_base_dir = Path('C:/Users/s.nakamura/workspace/projects/aicello/aug_test_data')
    for threshold in [160]:
        output_dir = output_base_dir / str(threshold)
        main(input_image_dir, input_image_dir, output_dir, threshold)
