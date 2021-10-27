import glob
import SimpleITK as sitk
import nibabel as nb
import os
import numpy as np
from scipy.interpolate import interpn

# path_list = glob.glob('E:/HSE/Thyroid/Dicom/*/crop_ct.nii.gz')
path_list = glob.glob('D:/0902_Thyroid/ThyroidSPECT Dataset/*/Tc Thyroid SPECT/crop_ct.nii.gz')
src_img = nb.load(path_list[0])
src_img_data = src_img.get_fdata()
print(f'affine = {src_img.affine}')
# nb.save(nb.Nifti1Pair(src_img_data, src_img.affine), 'affine_spect.nii.gz')


# src_img = sitk.ReadImage(path_list[0])
# src_img_data = sitk.GetArrayFromImage(src_img)
# affine = src_img.GetOrigin()
# print(f'origin = {affine}')


print(f'File saved in {os.getcwd()}')