import numpy as np
import SimpleITK as sitk
import os

# for i in reversed(range(3)):
#     print(i)

path = 'E:/HSE/Thyroid/Dicom/1.2.410.2000010.82.2291.2012505191227006/'

img_ct_path = path + 'crop_ct.nii.gz'
img_ct = sitk.ReadImage(img_ct_path)
# print(f'ct path for training = {img_ct_path}')
img_ct_data = sitk.GetArrayFromImage(img_ct)
img_ct_data = (img_ct_data - np.mean(img_ct_data)) / (np.std(img_ct_data) + 1e-8)

# For ConResNet
# img -> res
ct_size = np.shape(img_ct_data)[0]
ct_copy = np.zeros((ct_size, ct_size, ct_size)).astype(np.float32)
# print(f'ct_copy shape = {np.shape(ct_copy)}')
ct_copy[1:, :, :] = img_ct_data[0: ct_size - 1, :, :]
ct_res = img_ct_data - ct_copy
ct_res[0, :, :] = 0

ct_res_img = sitk.GetImageFromArray(ct_res)
sitk.WriteImage(ct_res_img[:, :, :], 'ct_res.nii.gz')
print(f'file saved in {os.getcwd()}')


mask_file = path + 'crop_mask.nii.gz'
print(f'mask file = {mask_file}')
mask_img = sitk.ReadImage(mask_file)
img_mask_data = sitk.GetArrayFromImage(mask_img)

# label -> res
# img_mask_data = np.reshape(img_mask_data, (1, ct_size, ct_size, ct_size))
mask_copy = np.zeros((64, 64, 64)).astype(np.float32)
mask_copy[1:, :, :] = img_mask_data[0: ct_size - 1, :, :]
mask_res = img_mask_data - mask_copy

mask_res_img = sitk.GetImageFromArray(mask_res)
sitk.WriteImage(mask_res_img[:, :, :], 'mask_res.nii.gz')

#
# mask_1 = np.where(mask_data == 7168, 1, 0)
# mask_2 = np.where(mask_data == 5120, 1, 0)
#
# mask1_nzero = mask_1.nonzero()
# mask2_nzero = mask_2.nonzero()
# # if not mask1_nzero:
# #     print(f'mask 1 empty')
# if not mask2_nzero[2].size > 0 or not mask1_nzero[2].size:
#     print(f'mask empty')
# # print(f'np.min(mask1_nzero[2]) = {mask1_nzero}, np.min(mask2_nzero[2]) = {mask2_nzero}')
#
#  A = np.array([[[ 1,  2,  3],[ 4,  5,  6],[12, 34, 90]],
#                 [[ 4,  5,  6],[ 2,  5,  6],[ 7,  3,  4]]])