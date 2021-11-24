import os
import nibabel as nb
import numpy as np
import tables
import SimpleITK as sitk
import glob
import torch

'''
Code to view the nii file in mricro with same origin/voxel size
Just Temporary
For Thyroid data
'''
# data_path = glob.glob('F:/LungCancerData/test/*/')
data_path = glob.glob('D:/0902_Thyroid/ThyroidSPECT Dataset/*/Tc Thyroid SPECT/')


def resave_file(file_path, save_path):
    file_name = file_path.split('\\')[-1]
    print(f'file name = {file_name}')
    src_img = sitk.ReadImage(file_path)
    # print(f'ct path for training = {img_ct_path}')
    src_arr = sitk.GetArrayFromImage(src_img)
    src_arr = np.where(src_arr > 0, 1, 0)
    src_img = sitk.GetImageFromArray(src_arr)
    os.chdir(save_path)
    sitk.WriteImage(src_img[:, :, :], 'crop_mask_compare.nii.gz')
    print(f'{file_name} saved in {os.getcwd()}')


for f in data_path:
    # ct_path = f + 'crop_ct.nii.gz'
    # if os.path.isfile(ct_path):
    #     resave_file(file_path=ct_path, save_path=f)
    # else:
    #     print('No thyroid ct file!')

    mask_path = f + 'crop_mask.nii.gz'
    if os.path.isfile(mask_path):
        resave_file(file_path=mask_path, save_path=f)
    else:
        print('No thyroid mask file!')

    # if os.path.isfile(roi_path):
    #     resave_file(file_path=roi_path, save_path=f)
    # else:
    #     print('No primary file!')
    # break