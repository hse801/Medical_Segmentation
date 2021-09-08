import os
import nibabel as nb
import numpy as np
import SimpleITK as sitk
import glob
import pydicom

file_path = glob.glob('E:/HSE/Thyroid/Dicom/*/')

print(f'len = {len(file_path)}')

for f in file_path:
    ct_list = glob.glob(f + 'CT*.dcm')
    print(f'len ct of {f} = {len(ct_list)}')