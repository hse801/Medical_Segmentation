"""
create flipped version of ct and mask file
to use for training the model
"""
import SimpleITK as sitk
import glob
import torch
import numpy as np
import os


file_list = glob.glob('E:/HSE/Thyroid/Dicom/*/')

for idx, path in enumerate(file_list):

    ct_file = path + 'crop_ct.nii.gz'
    img_ct = sitk.ReadImage(ct_file)
    # print(f'ct path for training = {img_ct_path}')
    img_ct_data = sitk.GetArrayFromImage(img_ct)
    img_ct_data = (img_ct_data - np.mean(img_ct_data)) / (np.std(img_ct_data) + 1e-8)

    flip_ct_data = img_ct_data[:, :, ::-1]
    print(f'bf flip ct data shape = {np.shape(flip_ct_data)}')
    # flip_ct_data = np.squeeze(flip_ct_data)
    # print(f'af flip ct data shape = {np.shape(flip_ct_data)}')
    os.chdir(path)
    flip_ct_img = sitk.GetImageFromArray(flip_ct_data)
    sitk.WriteImage(flip_ct_img[:, :, :], 'flip_ct.nii.gz')

    # flip mask file
    mask_file = path + 'crop_mask.nii.gz'
    print(f'mask file = {mask_file}')
    mask_img = sitk.ReadImage(mask_file)
    mask_data = sitk.GetArrayFromImage(mask_img)

