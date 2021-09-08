import glob
import SimpleITK as sitk
import nibabel as nb
import os
import numpy as np

file_path = glob.glob('D:/0902_Thyroid/ThyroidSPECT Dataset/*/*/')

for f in file_path:
    # print(f'f = {f}')
    ct_list = glob.glob(f + 'CTCT StandardThyroid SPECT/IM1.DCM')
    mask_list = glob.glob(f + 'Q.Metrix Organs/*.DCM')
    spect_list = glob.glob(f + 'Q.Metrix_Transaxials_IsotropNM/*.DCM')

    if len(ct_list) == 0:
        raise Exception('No CT files')

    if len(mask_list) != 1:
        raise Exception('Mask file # is not 1')

    if len(spect_list) != 1:
        raise Exception('Spect file # is not 1')

    # ct_list = glob.glob('D:/0902_Thyroid/ThyroidSPECT Dataset/10414125/*/CTCT StandardThyroid SPECT/IM1.DCM')
    # print(f'file list len = {len(ct_list)}')
    print(f'ct = {ct_list[0]} \nmask = {mask_list[0]} \nspect = {spect_list[0]}')
    os.chdir('C:/Users/Bohye/Desktop/mricron')

    os.system('dcm2nii.exe ' + ct_list[0])
    os.system('dcm2nii.exe ' + mask_list[0])
    os.system('dcm2nii.exe ' + spect_list[0])

    ct_nii = glob.glob(f + 'CTCT StandardThyroid SPECT/2*.nii.gz')
    mask_nii = glob.glob(f + 'Q.Metrix Organs/*.nii.gz')
    spect_nii = glob.glob(f + 'Q.Metrix_Transaxials_IsotropNM/*.nii.gz')


    break