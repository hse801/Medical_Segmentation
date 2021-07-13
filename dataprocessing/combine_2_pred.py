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
combine left and right prediction result into one nifti file
each from different model

'''

mask_list = glob.glob('E:/HSE/Thyroid/Dicom/*/')

for idx, path in enumerate(mask_list):
    left_file = glob.glob(path + 'pred_12_35_07_13*.nii.gz')
    right_file = glob.glob(path + 'pred_10_02_07*.nii.gz')

    left_img = sitk.ReadImage(left_file[0])
    left_data = sitk.GetArrayFromImage(left_img)

    right_img = sitk.ReadImage(right_file[0])
    right_data = sitk.GetArrayFromImage(right_img)

    combined_arr = left_data + right_data
    combined_arr[combined_arr > 1] = 1
    print(f'combined max = {np.max(combined_arr)}')

    os.chdir(path)
    combined_img = sitk.GetImageFromArray(combined_arr)
    sitk.WriteImage(combined_img[:, :, :], f'pred_2ch_combined_{idx}.nii.gz')
    print(f'combined file saved in {os.getcwd()}, idx = {idx}')
    # break