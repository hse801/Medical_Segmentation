import os

import nibabel as nib
import numpy as np
import tables
import SimpleITK as sitk
import glob
import torch
import lib.medloaders as dataloaders
import lib.medzoo as medzoo
from lib.losses3D import DiceLoss
from lib.visual3D_temp.BaseWriter import TensorboardWriter

folder_path = glob.glob('E:/HSE/Thyroid/Dicom/*/')

for idx, f in enumerate(folder_path):
    left_path = glob.glob(f + 'pred_00_35_07_26_left*')
    right_path = glob.glob(f + 'pred_11_00_07_29_right*')
    file_name = f'pred_0812_lr_sep_sum_{idx}.nii.gz'
    # print(file_name)

    img_left = sitk.ReadImage(left_path[0])
    img_left_data = sitk.GetArrayFromImage(img_left)

    img_right = sitk.ReadImage(right_path[0])
    img_right_data = sitk.GetArrayFromImage(img_right)

    # img_sum_data = img_right_data + img_left_data
    # img_sum_data = np.where(img_sum_data > 0, 1, 0)
    img_right_data = np.where(img_right_data > 0, 7168, 0)
    img_left_data = np.where(img_left_data > 0, 5120, 0)

    img_sum_data = img_right_data + img_left_data
    img_sum_data = np.where(img_sum_data > 7168, 7168, img_sum_data)

    sum_img = sitk.GetImageFromArray(img_sum_data[:, :, :])
    os.chdir(f)
    sitk.WriteImage(sum_img[:, :, :], file_name)
    print(f'{file_name} saved in {os.getcwd()}')
    # break
    if idx == 59:
        break
