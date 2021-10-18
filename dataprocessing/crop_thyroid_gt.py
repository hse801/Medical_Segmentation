import os
import nibabel as nb
import numpy as np
import tables
import SimpleITK as sitk
import glob
import torch
import lib.medloaders as dataloaders
import skimage.transform as skTrans


"""
Crop the 0902_Thyroid data
Before: 256 x 256 x 256
After: 64 x 64 x 64

Crop nii file based on ground truth file

"""


def crop_file(folder_path: str, count):
    ct_file = folder_path + 'CT_rsmpl.nii.gz'
    mask_file = folder_path + 'Mask_rsmpl.nii.gz'
    spect_file = folder_path + 'SPECT_rsmpl.nii.gz'

    img_ct = nb.load(ct_file)
    img_mask = nb.load(ct_file)
    img_spect = nb.load(ct_file)

    ct_arr = img_resize(ct_file)
    mask_arr = img_resize(mask_file)
    spect_arr = img_resize(spect_file)

    nzero = mask_arr.nonzero()

    x_mid = int((min(nzero[2]) + max(nzero[2])) / 2)
    y_mid = int((min(nzero[1]) + max(nzero[1])) / 2)
    z_mid = int((min(nzero[0]) + max(nzero[0])) / 2)
    print(f'xmid = {x_mid}, ymid = {y_mid}, zmid = {z_mid}')

    os.chdir(folder_path)

    cropped_ct_arr = crop_arr(x_mid, y_mid, z_mid, ct_arr)
    cropped_ct_img = nb.Nifti1Image(cropped_ct_arr, img_ct.affine, img_ct.header)
    nb.save(cropped_ct_img, f'crop_ct.nii.gz')

    cropped_mask_arr = crop_arr(x_mid, y_mid, z_mid, mask_arr)
    cropped_mask_img = nb.Nifti1Image(cropped_mask_arr, img_mask.affine, img_mask.header)
    nb.save(cropped_mask_img, f'crop_mask.nii.gz')

    cropped_spect_arr = crop_arr(x_mid, y_mid, z_mid, spect_arr)
    cropped_spect_img = nb.Nifti1Image(cropped_spect_arr, img_spect.affine, img_spect.header)
    nb.save(cropped_spect_img, f'crop_spect.nii.gz')

    print(f'files saved in {os.getcwd()}')


def img_resize(src_file: str, dim=128):
    # resize the file while maintaining the range
    # return numpy array after resizing

    img_src = nb.load(src_file)
    img_src_data = img_src.get_fdata()
    src_resize = skTrans.resize(img_src_data, (dim, dim, dim), order=1, preserve_range=True)
    # print(f'type = {type(src_resize)}, affine = {img_src.affine}, header = {img_src.header}')

    return src_resize


def crop_arr(x_mid, y_mid, z_mid, file_arr):
    # Get array as input
    # set the same voxel size with file before crop
    z_start, z_end = check_in_range(z_mid, crop_range=32, file_dim=256)
    y_start, y_end = check_in_range(y_mid, crop_range=32, file_dim=256)
    x_start, x_end = check_in_range(x_mid, crop_range=32, file_dim=256)
    cropped_arr = file_arr[z_start:z_end, y_start:y_end, x_start:x_end]

    return cropped_arr


def crop_file_to_img(x_mid, y_mid, z_mid, file_to_crop):

    file_img = nb.load(file_to_crop)
    file_arr = file_img.get_fdata()

    # set the same voxel size with file before crop
    z_start, z_end = check_in_range(z_mid, crop_range=32, file_dim=256)
    y_start, y_end = check_in_range(y_mid, crop_range=32, file_dim=256)
    x_start, x_end = check_in_range(x_mid, crop_range=32, file_dim=256)
    cropped_arr = file_arr[z_start:z_end, y_start:y_end, x_start:x_end]
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
count = 0

for i in folder_path:
    crop_file(i, count)
    count += 1
    print(f'count = {count}')
    print('-------------------------------\n')
    # break
