#!/bin/bash
# Download COCO128 calibration dataset (128 images, about 7 MB)
python3 -c "
from ultralytics.data.utils import check_det_dataset
from pathlib import Path
import shutil

src = Path(check_det_dataset('coco128.yaml')['train'])
dst = Path('calibration_images')
dst.mkdir(exist_ok=True)
for img in src.glob('*'):
    shutil.copy2(img, dst)
print(f'{len(list(dst.iterdir()))} images -> {dst}/')
"
