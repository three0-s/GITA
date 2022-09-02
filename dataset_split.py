import os
import glob
import shutil
import numpy as np

total_files = glob.glob('/home/yewon/GITA/dataset/train/*.png')
test_val_files = np.random.choice(total_files, 4096, replace=False)
test_files, val_files = test_val_files[:2048], test_val_files[2048:]

for test in test_files:
    dest = test.replace('train', 'test')
    shutil.move(test, dest)

for val in val_files:
    dest = val.replace('train', 'val')
    shutil.move(val, dest)
