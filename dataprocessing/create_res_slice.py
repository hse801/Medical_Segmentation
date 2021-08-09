import numpy as np
import SimpleITK as sitk
import os

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


mask_path = lung_path + 'ROI_cut.nii.gz'
# mask_file = path + 'crop_mask.nii.gz'
print(f'mask file = {mask_path}')
mask_img = sitk.ReadImage(mask_path)
img_mask_data = sitk.GetArrayFromImage(mask_img)

# label -> res
# img_mask_data = np.reshape(img_mask_data, (1, ct_size, ct_size, ct_size))
mask_copy = np.zeros((80, 128, 160)).astype(np.float32)
mask_copy[1:, :, :] = img_mask_data[0: ct_size - 1, :, :]
mask_res = img_mask_data - mask_copy
os.chdir(lung_path)
mask_res_img = sitk.GetImageFromArray(mask_res)
sitk.WriteImage(mask_res_img[:, :, :], 'mask_res.nii.gz')