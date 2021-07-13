import numpy as np
import SimpleITK as sitk

path = 'E:/HSE/Thyroid/Dicom/1.2.410.2000010.82.2291.2012505191227006/'

mask_file = path + 'crop_mask.nii.gz'
print(f'mask file = {mask_file}')
mask_img = sitk.ReadImage(mask_file)
mask_data = sitk.GetArrayFromImage(mask_img)

mask_1 = np.where(mask_data == 7168, 1, 0)
mask_2 = np.where(mask_data == 5120, 1, 0)

mask1_nzero = mask_1.nonzero()
mask2_nzero = mask_2.nonzero()
# if not mask1_nzero:
#     print(f'mask 1 empty')
if not mask2_nzero[2].size > 0 or not mask1_nzero[2].size:
    print(f'mask empty')
# print(f'np.min(mask1_nzero[2]) = {mask1_nzero}, np.min(mask2_nzero[2]) = {mask2_nzero}')

 A = np.array([[[ 1,  2,  3],[ 4,  5,  6],[12, 34, 90]],
                [[ 4,  5,  6],[ 2,  5,  6],[ 7,  3,  4]]])