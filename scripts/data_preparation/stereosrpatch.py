import os
import glob
from shutil import copy
import sys
import cv2

scale = 4
input_dir = '/media/alex/data/NAFNet/datasets/Train/HR'
output_dir = '/media/alex/data/NAFNet/datasets/Train/pacthes_x4'
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
img_list_l = sorted(glob.glob(os.path.join(input_dir, '*_L.png')))
img_list_r = sorted(glob.glob(os.path.join(input_dir, '*_R.png')))

for i in range(len(img_list_l)):
    img_0 = cv2.imread(img_list_l(i))
    img_1 = cv2.imread(img_list_r(i))

    H, W, C = img_0.shape
    img_hr_0 = img_0[:H//scale * scale,:W//scale * scale,:]
    img_hr_1 = img_1[:H // scale * scale, :W // scale * scale, :]
    img_lr_0 = cv2.resize()


for idx, file in enumerate(img_list_l):
    save_dir = os.path.join(output_dir, str(idx+1))
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
