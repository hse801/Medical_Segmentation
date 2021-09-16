import numpy as np
import SimpleITK as sitk
import os
import scipy
import scipy.ndimage as ndimage
import torchio as tio

# For ConResNet
# img -> res
lung_path = 'E:/HSE/LungCancerData/train/27903971/'
# ct_path = lung_path + 'CT_cut.nii.gz'
ct_path = lung_path + 'CT_cut.nii.gz'
lymph_path = lung_path + 'lymph_cut_sum.nii.gz'

os.chdir(lung_path)
img_ct = sitk.ReadImage(ct_path)
img_ct_data = sitk.GetArrayFromImage(img_ct)
img_ct_data = (img_ct_data - np.mean(img_ct_data)) / (np.std(img_ct_data) + 1e-8)
ct_size = np.shape(img_ct_data)[0]
# ct_copy = np.zeros((80, 128, 160)).astype(np.float32)
ct_copy = np.zeros((80, 128, 160)).astype(np.float32)
# print(f'ct_copy shape = {np.shape(ct_copy)}')
ct_copy[1:, :, :] = img_ct_data[0: ct_size - 1, :, :]
ct_res = img_ct_data - ct_copy
ct_res[0, :, :] = 0

# sx = ndimage.sobel(img_ct_data, axis=0, mode='constant')
# sy = ndimage.sobel(img_ct_data, axis=1, mode='constant')
# ct_res = np.hypot(sx, sy)

ct_res_img = sitk.GetImageFromArray(ct_res)
sitk.WriteImage(ct_res_img[:, :, :], 'CT_res.nii.gz')
print(f'file saved in {os.getcwd()}')


# mask_path = lung_path + 'ROI_cut.nii.gz'
mask_path = lung_path + 'ROI_cut.nii.gz'
# mask_file = path + 'crop_mask.nii.gz'
print(f'mask file = {mask_path}')
mask_img = sitk.ReadImage(mask_path)
img_mask_data = sitk.GetArrayFromImage(mask_img)

img_lymph = sitk.ReadImage(lymph_path)
img_lymph_data = sitk.GetArrayFromImage(img_lymph)

img_mask_data = img_mask_data + img_lymph_data
img_mask_data[img_mask_data > 0] = 1

# label -> res
# img_mask_data = np.reshape(img_mask_data, (1, ct_size, ct_size, ct_size))
mask_copy = np.zeros((80, 128, 160)).astype(np.float32)
mask_copy[1:, :, :] = img_mask_data[0: ct_size - 1, :, :]
mask_res = img_mask_data - mask_copy
# zoom_percentage = 0.9
# zoom_matrix = np.array([[zoom_percentage, 0, 0, 0],
#                         [0, zoom_percentage, 0, 0],
#                         [0, 0, zoom_percentage, 0],
#                         [0, 0, 0, 1]])
# mask_res = ndimage.interpolation.affine_transform(img_mask_data, zoom_matrix)

# Apply Sobel Filter
# sx = ndimage.sobel(img_mask_data, axis=1, mode='constant')
# sy = ndimage.sobel(img_mask_data, axis=2, mode='constant')
# print(f'shape sx = {len(sx)}, sy = {len(sy)}')
# mask_res = np.hypot(sx, sy)
# mask_res = (mask_res - np.min(mask_res)) / (np.max(mask_res) - np.min(mask_res))
# mask_res = sy
# mask_res = np.where(mask_res > 0.5, 1, 0)
# mask_res[mask_res > 0] = 1
# print(f'edge = {type(edge)}, shape = {np.shape(edge)}')

# mask_res = scipy.ndimage.interpolation.zoom(img_mask_data, 0.9)
# mask_res = img_mask_data - mask_copy
# image = tio.ScalarImage(mask_path)
# transform = tio.CropOrPad((58, 58, 58))
# output = transform(image)
# print(f'{type(output)}')
os.chdir(lung_path)
mask_res_img = sitk.GetImageFromArray(mask_res)
sitk.WriteImage(mask_res_img[:, :, :], 'MASK_res.nii.gz')
print(f'Mask res file saved in {os.getcwd()}')
