import glob
import SimpleITK as sitk
import nibabel as nb
import os
import numpy as np
from scipy.interpolate import interpn
import itertools
import skimage.transform as skTrans

"""
For thyroid_0902 data
1. saved .hdr and .img file
convert to nii.gz file format(256x256x~)
2. crop the image using gt image(128x128x128)
3. resize the image(64x64x64)
"""


def hdr_to_nii(path: str):

    files = glob.glob(path + '*.img')
    os.chdir(path)
    for f in files:
        img = nb.load(f)
        nb.save(img, f.replace('.img', '.nii.gz'))
    print(f'{len(files)} files saved in {os.getcwd()}')


def nii_resize(img_src_data: str):

    new_size_x = 64
    new_size_y = 64
    new_size_z = 64

    # img_src = nb.load(src_file)
    # img_src_data = img_src.get_fdata()

    # initial_size_x = np.shape(img_src_data)[0]
    # initial_size_y = np.shape(img_src_data)[1]
    # initial_size_z = np.shape(img_src_data)[2]
    initial_size_x = np.shape(img_src_data)[0]
    initial_size_y = np.shape(img_src_data)[0]
    initial_size_z = np.shape(img_src_data)[0]

    delta_x = initial_size_x/new_size_x
    delta_y = initial_size_y/new_size_y
    delta_z = initial_size_z/new_size_z

    new_data = np.zeros((new_size_x,new_size_y,new_size_z))

    for x, y, z in itertools.product(range(new_size_x),
                                     range(new_size_y),
                                     range(new_size_z)):
        new_data[x][y][z] = img_src_data[int(x*delta_x)][int(y*delta_y)][int(z*delta_z)]

    # img = nb.Nifti1Image(new_data, np.eye(4))
    # img.to_filename(file_name)
    # print(f'Resized file {file_name} is saved in {os.getcwd()}')
    return new_data


def crop_file(folder_path: str):
    ct_file = folder_path + 'CT_rsmpl_0902.nii.gz'
    mask_file = folder_path + 'Mask_rsmpl_0902.nii.gz'
    spect_file = folder_path + 'SPECT_0902.nii.gz'

    img_ct = nb.load(ct_file)
    img_mask = nb.load(mask_file)
    img_spect = nb.load(spect_file)

    ct_arr = img_ct.get_fdata()
    mask_arr = img_mask.get_fdata()
    spect_arr = img_spect.get_fdata()

    # ct_arr = img_resize(ct_file)
    # mask_arr = img_resize(mask_file)
    # spect_arr = img_resize(spect_file)

    nzero = mask_arr.nonzero()

    x_mid = int((min(nzero[2]) + max(nzero[2])) / 2)
    y_mid = int((min(nzero[1]) + max(nzero[1])) / 2)
    z_mid = int((min(nzero[0]) + max(nzero[0])) / 2)
    print(f'xmid = {x_mid}, ymid = {y_mid}, zmid = {z_mid}')

    os.chdir(folder_path)

    cropped_ct_arr = crop_arr(x_mid, y_mid, z_mid, ct_arr)
    cropped_ct_arr = nii_resize(img_src_data=cropped_ct_arr)
    cropped_ct_img = nb.Nifti1Image(cropped_ct_arr, img_ct.affine, img_ct.header)
    nb.save(cropped_ct_img, f'crop_ct.nii.gz')

    cropped_mask_arr = crop_arr(x_mid, y_mid, z_mid, mask_arr)
    cropped_mask_arr = nii_resize(img_src_data=cropped_mask_arr)
    cropped_mask_img = nb.Nifti1Image(cropped_mask_arr, img_mask.affine, img_mask.header)
    nb.save(cropped_mask_img, f'crop_mask.nii.gz')

    cropped_spect_arr = crop_arr(x_mid, y_mid, z_mid, spect_arr)
    cropped_spect_arr = nii_resize(img_src_data=cropped_spect_arr)
    cropped_spect_img = nb.Nifti1Image(cropped_spect_arr, img_spect.affine, img_spect.header)
    nb.save(cropped_spect_img, f'crop_spect.nii.gz')

    print(f'files saved in {os.getcwd()}')


def crop_arr(x_mid, y_mid, z_mid, file_arr):
    # Get array as input
    # set the same voxel size with file before crop
    z_start, z_end = check_in_range(z_mid, crop_range=64, file_dim=256)
    y_start, y_end = check_in_range(y_mid, crop_range=64, file_dim=256)
    x_start, x_end = check_in_range(x_mid, crop_range=64, file_dim=256)
    cropped_arr = file_arr[z_start:z_end, y_start:y_end, x_start:x_end]

    return cropped_arr


def check_in_range(mid, crop_range, file_dim):
    if mid + crop_range > file_dim:
        start = file_dim - crop_range * 2
        end = file_dim
        print('range over max')
    elif mid - crop_range < 0:
        start = 0
        end = crop_range * 2
        print(f'range under min in {os.getcwd()}')
    else:
        start = mid - crop_range
        end = mid + crop_range

    return start, end

data_path = glob.glob('D:/0902_Thyroid/ThyroidSPECT Dataset/*/Tc Thyroid SPECT/')

for p in data_path:
    hdr_to_nii(path=p)
    ct_file = p + '/CT_rsmpl_0902.nii.gz'
    spect_file = p + '/SPECT_0902.nii.gz'
    mask_file = p + '/Mask_rsmpl_0902.nii.gz'
    crop_file(folder_path=p)
    # nii_resize(src_file=ct_file, file_name='CT_resize_0902.nii.gz')
    # nii_resize(src_file=spect_file, file_name='SPECT_resize_0902.nii.gz')
    # nii_resize(src_file=mask_file, file_name='MASK_resize_0902.nii.gz')
    # break
