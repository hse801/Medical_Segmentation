import os
import glob


folder_path = glob.glob('E:/HSE/Thyroid/Dicom/*/test*.nii.gz')
count = 0

for i in folder_path:
    os.remove(i)
    count += 1
    print(f'{i} is removed from {os.getcwd()}')
    print('count = ', count)
