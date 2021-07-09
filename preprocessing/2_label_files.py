import os
import nibabel as nb
import numpy as np
import tables
import SimpleITK as sitk
import glob
import torch
import lib.medloaders as dataloaders
import lib.medzoo as medzoo
from lib.losses3D import DiceLoss

'''
create two mask data files for left and right thyroid

'''

mask_list = glob.glob('E:/HSE/Thyroid/Dicom/*/')

for idx, path in enumerate(mask_list):

    mask_file = path + 'crop_mask.nii.gz'
    print(f'mask file = {mask_file}')
    mask_img = sitk.ReadImage(mask_file)
    mask_data = sitk.GetArrayFromImage(mask_img)

    mask_left = np.where(mask_data == 5120, 1, 0)
    mask_right = np.where(mask_data == 7168, 1, 0)

    mask_left_img = sitk.GetImageFromArray(mask_left[:, :, :])
    mask_right_img = sitk.GetImageFromArray(mask_right[:, :, :])

    os.chdir(path)
    sitk.WriteImage(mask_left_img[:, :, :], 'crop_mask_left.nii.gz')
    sitk.WriteImage(mask_right_img[:, :, :], 'crop_mask_right.nii.gz')
    print(f'file saved in {os.getcwd()}')
    # break


# mask_list = glob.glob('E:/HSE/Thyroid/Dicom/*/')
# for i in mask_list:
#     get_2_labels(i)
#     break