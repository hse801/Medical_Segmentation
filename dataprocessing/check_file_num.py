import glob
import os

ct_path = glob.glob('E:/HSE/LungCancerData/train/*/CT_cut.nii.gz')
pet_path = glob.glob('E:/HSE/LungCancerData/train/*/PET_cut.nii.gz')
primary_path = glob.glob('E:/HSE/LungCancerData/train/*/ROI_cut.nii.gz')

print(f'ct_path = {len(ct_path)}, pet_path = {len(pet_path)}, primary_path = {len(primary_path)}')