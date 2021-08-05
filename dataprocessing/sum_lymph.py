import os
import glob
import numpy as np
import SimpleITK as sitk
import nibabel as nb
import itk

"""
every patients have different numbers of lymph node data
collect all data, sum and save in nii.gz 
for convenience

use nibabel library to save affine information together
(voxel size, origin)
"""

folder_path = glob.glob('E:/HSE/LungCancerData/test/*/')

# With Nibabel
for i in folder_path:
    lymph_path = glob.glob(i + '*lymph_cut.nii.gz')
    # sample_img = nb.load(lymph_path[0])
    sum_lymph = np.zeros((160, 128, 80))
    # sum_lymph = nb.Nifti1Image(zero_arr, sample_img.affine)
    for l in lymph_path:
        print(f'lymph file = {l}')
        lymph_img = nb.load(l)
        lymph_affine = lymph_img.affine
        # print(f'lymph_affine = {lymph_affine}')
        lymph_data = lymph_img.get_fdata()
        # print(f'type sum_lymph = {sum_lymph.shape}, type lymph_img = {lymph_img.shape}')
        # type sum_lymph = (160, 128, 80), type lymph_img = (160, 128, 80)
        sum_lymph += lymph_data

    sum_lymph[sum_lymph > 1] = 1
    file_name = 'lymph_cut_sum.nii.gz'
    os.chdir(i)
    if len(lymph_path) != 0:
        nb.Nifti1Image(sum_lymph, lymph_affine).to_filename(file_name)
        print(f'{file_name} saved in {os.getcwd()}')
    else:
        print(f'No lymph node in {os.getcwd()}')
    # break

# WIth SimpleITK
# for i in folder_path:
#     lymph_path = glob.glob(i + '*lymph_cut.nii.gz')
#     zero_arr = np.zeros((80, 128, 160))
#     sum_lymph = sitk.GetImageFromArray(zero_arr)
#     for l in lymph_path:
#         print(f'lymph file = {l}')
#         lymph_img = sitk.ReadImage(l)
#         lymph_origin = lymph_img.GetOrigin()
#         print(f'lymph_origin = {lymph_origin}')
#         # lymph_img = lymph_img[::-1, :, :]
#         lymph_data = sitk.GetArrayFromImage(lymph_img)
#         print(f'type sum_lymph = {sum_lymph.GetSize()}, type lymph_img = {lymph_img.GetSize()}')
#         sum_lymph += lymph_img
#
#         sum_lymph[sum_lymph > 1] = 1
#     file_name = 'lymph_cut_sum.nii.gz'
#     os.chdir(i)
#     if len(lymph_path) != 0:
#         sum_lymph_img = sitk.GetImageFromArray(sum_lymph)
#         # sitk.WriteImage(sum_lymph_img[:, :, :], file_name)
#         # sitk.WriteImage(sum_lymph[:, :, :], file_name)
#         print(f'{file_name} saved in {os.getcwd()}')
#     else:
#         print(f'No lymph node in {os.getcwd()}')
#     # break
#     # print(f'lymph path = {i}')


# def collect_lymph(lymph_path):
#     patient