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
from lib.visual3D_temp.BaseWriter import TensorboardWriter

"""
Crop the 0902_Thyroid data
Before: 256 x 256 x 256
After: 64 x 64 x 64

Crop nii file based on ground truth file

"""


def crop_file(folder_path, count):
    ct_file = folder_path + 'CT_rsmpl.nii.gz'
    mask_file = folder_path + 'Mask_rsmpl.nii.gz'
    spect_file = folder_path + 'SPECT_rsmpl.nii.gz'
    # pred_file = glob.glob(folder_path + f'pred_{count}.nii.gz')

    img_mask = nb.load(mask_file)
    img_mask_data = img_mask.get_fdata()
    nzero = img_mask_data.nonzero()
    # print(f'nzero shape = {np.shape(nzero)}')
    # print(f'nzero = {nzero}')
    # print(f'nzero[0] = {min(nzero[0])}')
    # print(f'nzero[2] = {nzero}')
    x_mid = int((min(nzero[2]) + max(nzero[2])) / 2)
    y_mid = int((min(nzero[1]) + max(nzero[1])) / 2)
    z_mid = int((min(nzero[0]) + max(nzero[0])) / 2)
    print(f'xmid = {x_mid}, ymid = {y_mid}, zmid = {z_mid}')

    os.chdir(folder_path)

    cropped_ct_img = crop_file_to_img(x_mid, y_mid, z_mid, ct_file)
    nb.save(cropped_ct_img, f'crop_ct.nii.gz')

    cropped_spect_img = crop_file_to_img(x_mid, y_mid, z_mid, spect_file)
    nb.save(cropped_spect_img, f'crop_spect.nii.gz')

    cropped_mask_img = crop_file_to_img(x_mid, y_mid, z_mid, mask_file)
    nb.save(cropped_mask_img, f'crop_mask.nii.gz')

    print(f'files saved in {os.getcwd()}')
    # x_max =


def crop_file_to_img(x_mid, y_mid, z_mid, file_to_crop):

    file_img = nb.load(file_to_crop)
    file_arr = file_img.get_fdata()

    # set the same voxel size with file before crop
    # print(f'file_spacing = {file_spacing}, file_origin = {file_origin}, file_direction = {file_direction}')
    z_start, z_end = check_in_range(z_mid, crop_range=32, file_dim=256)
    y_start, y_end = check_in_range(y_mid, crop_range=32, file_dim=256)
    x_start, x_end = check_in_range(x_mid, crop_range=32, file_dim=256)
    cropped_arr = file_arr[z_start:z_end, y_start:y_end, x_start:x_end]
    # print(f'file_img.affine = {file_img.affine}')
    cropped_img = nb.Nifti1Image(cropped_arr, file_img.affine, file_img.header)

    return cropped_img


def check_in_range(mid, crop_range, file_dim):
    if mid + crop_range > file_dim:
        start = file_dim - crop_range * 2
        end = file_dim
        print('range over max')
    elif mid - crop_range < 0:
        start = 0
        end = crop_range * 2
        print('range under min')
    else:
        start = mid - crop_range
        end = mid + crop_range

    return start, end


_, _, pred_loader = dataloaders.thyroid_dataloader.generate_thyroid_dataset()

folder_path = glob.glob('D:/0902_Thyroid/ThyroidSPECT Dataset/*/Tc Thyroid SPECT/')
# print(f'type = {type(folder_path)}, len = {len(folder_path)}')
count = 0
for i in folder_path:
    crop_file(i, count)
    count += 1
    print(f'count = {count}')
    print('-------------------------------\n')
    # break
# ct_path = glob.glob('E:/HSE/Thyroid/Dicom/*/CT_rsmpl.nii.gz')
# mask_path = glob.glob('E:/HSE/Thyroid/Dicom/*/Mask_rsmpl.nii.gz')