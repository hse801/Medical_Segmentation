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
'''
data_path = glob.glob('F:/LungCancerData/valid/*/')


def resave_file(file_path, save_path):
    file_name = file_path.split('\\')[-1]
    print(f'file name = {file_name}')
    src_img = sitk.ReadImage(file_path)
    # print(f'ct path for training = {img_ct_path}')
    src_arr = sitk.GetArrayFromImage(src_img)
    src_img = sitk.GetImageFromArray(src_arr)
    os.chdir(save_path)
    sitk.WriteImage(src_img[:, :, :], file_name)
    print(f'{file_name} saved in {os.getcwd()}')


for f in data_path:
    lymph_path = f + 'lymph_cut_sum.nii.gz'
    roi_path = f + 'ROI_cut.nii.gz'
    if os.path.isfile(lymph_path):
        resave_file(file_path=lymph_path, save_path=f)
    else:
        print('No lymph file!')
    if os.path.isfile(roi_path):
        resave_file(file_path=roi_path, save_path=f)
    else:
        print('No primary file!')
    # break