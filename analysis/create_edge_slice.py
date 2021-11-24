import numpy as np
import SimpleITK as sitk
import os
import scipy
import scipy.ndimage as ndimage
import torchio as tio
import glob
from torch.nn import functional as F

"""
create edge image for thyroid data
1. inter-slice
2. sobel filter
"""
# img -> res
path_list = glob.glob('D:/0902_Thyroid/ThyroidSPECT Dataset/*/Tc Thyroid SPECT/')


def get_sobel_edge(src_file: str, file_name: str):
    src_img = sitk.ReadImage(src_file)
    img_src_data = sitk.GetArrayFromImage(src_img)
    img_src_data = img_src_data
    img_src_data[img_src_data > 0] = 1

    sx = ndimage.sobel(img_src_data, axis=0, mode='constant')
    sy = ndimage.sobel(img_src_data, axis=1, mode='constant')
    # print(f'shape sx = {len(sx)}, sy = {len(sy)}')
    src_edge = np.hypot(sx, sy)
    src_edge_img = sitk.GetImageFromArray(src_edge)
    sitk.WriteImage(src_edge_img[:, :, :], file_name)
    print(f'Sobel edge {file_name} saved in {os.getcwd()}')


def get_inter_slice_edge(src_file: str, file_name: str):
    src_img = sitk.ReadImage(src_file)
    img_src_data = sitk.GetArrayFromImage(src_img)
    img_src_data = img_src_data
    img_src_data[img_src_data > 0] = 1

    src_copy = np.zeros((64, 64, 64)).astype(np.float32)
    src_copy[1:, :, :] = img_src_data[0: 64 - 1, :, :]
    src_edge = img_src_data - src_copy

    src_edge_img = sitk.GetImageFromArray(src_edge)
    sitk.WriteImage(src_edge_img[:, :, :], file_name)
    print(f'Inter slice edge {file_name} saved in {os.getcwd()}')


for path in path_list:
    mask_file = path + 'crop_mask.nii.gz'
    # pet_path = path + 'PET_cut.nii.gz'
    os.chdir(path)
    get_sobel_edge(src_file=mask_file, file_name='sobel_edge.nii.gz')
    get_inter_slice_edge(src_file=mask_file, file_name='inter_slice_edge.nii.gz')
    break


