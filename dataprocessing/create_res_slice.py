import numpy as np
import SimpleITK as sitk
import os
import scipy
import scipy.ndimage as ndimage
import torchio as tio

# For ConResNet
# img -> res
# lung_path = 'E:/HSE/LungCancerData/train/27903971/'
# ct_path = lung_path + 'CT_cut.nii.gz'
thyroid_path = 'E:/HSE/Thyroid/Dicom/1.2.410.2000010.82.2291.1004615180322001.99369/'
ct_path = thyroid_path + 'crop_ct.nii.gz'

os.chdir(thyroid_path)
img_ct = sitk.ReadImage(ct_path)
img_ct_data = sitk.GetArrayFromImage(img_ct)
img_ct_data = (img_ct_data - np.mean(img_ct_data)) / (np.std(img_ct_data) + 1e-8)
ct_size = np.shape(img_ct_data)[0]
# ct_copy = np.zeros((80, 128, 160)).astype(np.float32)
ct_copy = np.zeros((64, 64, 64)).astype(np.float32)
# print(f'ct_copy shape = {np.shape(ct_copy)}')
ct_copy[1:, :, :] = img_ct_data[0: ct_size - 1, :, :]
ct_res = img_ct_data - ct_copy
ct_res[0, :, :] = 0

sx = ndimage.sobel(img_ct_data, axis=0, mode='constant')
sy = ndimage.sobel(img_ct_data, axis=1, mode='constant')
ct_res = np.hypot(sx, sy)

ct_res_img = sitk.GetImageFromArray(ct_res)
sitk.WriteImage(ct_res_img[:, :, :], 'ct_res2.nii.gz')
print(f'file saved in {os.getcwd()}')


# mask_path = lung_path + 'ROI_cut.nii.gz'
mask_path = thyroid_path + 'crop_mask.nii.gz'
# mask_file = path + 'crop_mask.nii.gz'
print(f'mask file = {mask_path}')
mask_img = sitk.ReadImage(mask_path)
img_mask_data = sitk.GetArrayFromImage(mask_img)


# label -> res
# img_mask_data = np.reshape(img_mask_data, (1, ct_size, ct_size, ct_size))
mask_copy = np.zeros((64, 64, 64)).astype(np.float32)
# mask_copy[3:, :, :] = img_mask_data[0: ct_size - 3, :, :]
# zoom_percentage = 0.9
# zoom_matrix = np.array([[zoom_percentage, 0, 0, 0],
#                         [0, zoom_percentage, 0, 0],
#                         [0, 0, zoom_percentage, 0],
#                         [0, 0, 0, 1]])
# mask_res = ndimage.interpolation.affine_transform(img_mask_data, zoom_matrix)
sx = ndimage.sobel(img_mask_data, axis=0, mode='constant')
sy = ndimage.sobel(img_mask_data, axis=1, mode='constant')
mask_res = np.hypot(sx, sy)

# print(f'edge = {type(edge)}, shape = {np.shape(edge)}')

# mask_res = scipy.ndimage.interpolation.zoom(img_mask_data, 0.9)
# mask_res = img_mask_data - mask_copy
# image = tio.ScalarImage(mask_path)
# transform = tio.CropOrPad((58, 58, 58))
# output = transform(image)
# print(f'{type(output)}')
os.chdir(thyroid_path)
mask_res_img = sitk.GetImageFromArray(mask_res)
sitk.WriteImage(mask_res_img[:, :, :], 'mask_res2.nii.gz')
