import os
import glob
from shutil import copy
import sys
import cv2

input_dir = '/media/alex/data/NAFNet/datasets/Train/HR'
output_dir = '/media/alex/data/NAFNet/datasets/Train/hr'
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
img_list_l = sorted(glob.glob(os.path.join(input_dir, '*_L.png')))
img_list_r = sorted(glob.glob(os.path.join(input_dir, '*_R.png')))

for idx, file in enumerate(img_list_l):
    save_dir = os.path.join(output_dir, str(idx+1))
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    copy(file, os.path.join(save_dir, 'hr0.png'))
    copy(img_list_r[idx], os.path.join(save_dir, 'hr1.png'))