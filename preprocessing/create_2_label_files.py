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

라벨의 좌우가 바뀐 경우 있음
nonzero 영역의 인덱스를 비교하여 라벨 지정
'''

mask_list = glob.glob('E:/HSE/Thyroid/Dicom/*/')

for idx, path in enumerate(mask_list):

    mask_file = path + 'crop_mask.nii.gz'
    print(f'mask file = {mask_file}')
    mask_img = sitk.ReadImage(mask_file)
    mask_data = sitk.GetArrayFromImage(mask_img)

    mask_1 = np.where(mask_data == 5120, 1, 0)
    mask_2 = np.where(mask_data == 7168, 1, 0)

    mask1_nzero = mask_1.nonzero()
    mask2_nzero = mask_2.nonzero()
    print(f'np.min(mask1_nzero[2]) = {np.min(mask1_nzero[2])}, np.min(mask2_nzero[2]) = {np.min(mask2_nzero[2])}')

    # if mask has only left or right thyroid
    if not mask2_nzero[2].size > 0 or not mask1_nzero[2].size:
        mask_left = mask_1
        mask_right = mask_2
    # figure which array is left and right
    if np.min(mask1_nzero[2]) < np.min(mask2_nzero[2]):
        mask_left = mask_1
        mask_right = mask_2
    else:
        mask_left = mask_2
        mask_right = mask_1

    mask_left_img = sitk.GetImageFromArray(mask_left[:, :, :])
    mask_right_img = sitk.GetImageFromArray(mask_right[:, :, :])

    os.chdir(path)
    sitk.WriteImage(mask_left_img[:, :, :], 'crop_mask_left.nii.gz')
    sitk.WriteImage(mask_right_img[:, :, :], 'crop_mask_right.nii.gz')
    print(f'file saved in {os.getcwd()}')
    print(f'idx = {idx}')
    # break


# mask_list = glob.glob('E:/HSE/Thyroid/Dicom/*/')
# for i in mask_list:
#     get_2_labels(i)
#     break