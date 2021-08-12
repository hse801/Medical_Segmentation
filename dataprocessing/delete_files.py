import os
import glob


# folder_path = glob.glob('E:/HSE/Thyroid/Dicom/*/test*.nii.gz')
# folder_path = glob.glob('E:/HSE/LungCancerData/train/*/lymph_cut_sum.nii.gz')
folder_path = glob.glob('E:/HSE/Thyroid/Dicom/*/pred_11_00_07_29*')
count = 0

for i in folder_path:
    os.remove(i)
    count += 1
    print(f'{i} is removed from {os.getcwd()}')
    print('count = ', count)
