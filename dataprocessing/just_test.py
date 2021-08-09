import numpy as np
import SimpleITK as sitk
import os

# for i in reversed(range(3)):
#     print(i)

# path = 'E:/HSE/Thyroid/Dicom/1.2.410.2000010.82.2291.2012505191227006/'
#
# img_ct_path = path + 'crop_ct.nii.gz'
# img_ct = sitk.ReadImage(img_ct_path)
# # print(f'ct path for training = {img_ct_path}')
# img_ct_data = sitk.GetArrayFromImage(img_ct)
# img_ct_data = (img_ct_data - np.mean(img_ct_data)) / (np.std(img_ct_data) + 1e-8)
#
# flip_ct_data = img_ct_data[:, :, ::-1]
# print(f'bf flip ct data shape = {np.shape(flip_ct_data)}')
# # flip_ct_data = np.squeeze(flip_ct_data)
# # print(f'af flip ct data shape = {np.shape(flip_ct_data)}')
# os.chdir(path)
# flip_ct_img = sitk.GetImageFromArray(flip_ct_data)
# sitk.WriteImage(flip_ct_img[:, :, :], 'flip_ct.nii.gz')
#
# print(f'file saved in {os.getcwd()}')

# For ConResNet
# img -> res
lung_path = 'E:/HSE/LungCancerData/train/27903971/'
ct_path = lung_path + 'CT_cut.nii.gz'
os.chdir(lung_path)
img_ct = sitk.ReadImage(ct_path)
img_ct_data = sitk.GetArrayFromImage(img_ct)
img_ct_data = (img_ct_data - np.mean(img_ct_data)) / (np.std(img_ct_data) + 1e-8)
ct_size = np.shape(img_ct_data)[0]
ct_copy = np.zeros((80, 128, 160)).astype(np.float32)
# print(f'ct_copy shape = {np.shape(ct_copy)}')
ct_copy[1:, :, :] = img_ct_data[0: ct_size - 1, :, :]
ct_res = img_ct_data - ct_copy
ct_res[0, :, :] = 0

ct_res_img = sitk.GetImageFromArray(ct_res)
sitk.WriteImage(ct_res_img[:, :, :], 'ct_res.nii.gz')
print(f'file saved in {os.getcwd()}')
#
#
mask_path = lung_path + 'ROI_cut.nii.gz'
# mask_file = path + 'crop_mask.nii.gz'
print(f'mask file = {mask_path}')
mask_img = sitk.ReadImage(mask_path)
img_mask_data = sitk.GetArrayFromImage(mask_img)
# flip_mask_data = img_mask_data[:, :, ::-1]
#
# flip_mask_img = sitk.GetImageFromArray(flip_mask_data)
# sitk.WriteImage(flip_mask_img[:, :, :], 'flip_mask.nii.gz')

#
# # label -> res
# # img_mask_data = np.reshape(img_mask_data, (1, ct_size, ct_size, ct_size))
mask_copy = np.zeros((80, 128, 160)).astype(np.float32)
mask_copy[1:, :, :] = img_mask_data[0: ct_size - 1, :, :]
mask_res = img_mask_data - mask_copy
os.chdir(lung_path)
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