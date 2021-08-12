import os
import nibabel as nib
import numpy as np
import tables
import SimpleITK as sitk
import glob
import torch

folder_path = glob.glob('E:/HSE/Thyroid/Dicom/*/')

for idx, f in enumerate(folder_path):
    gt_path = glob.glob(f + 'pred_0812_lr_sep_sum*')
    pred_path = glob.glob(f + 'crop_mask.nii*')
    file_name = f'compare_0812_{idx}.nii.gz'

    img_gt = sitk.ReadImage(gt_path[0])
    img_gt_data = sitk.GetArrayFromImage(img_gt)

    img_pred = sitk.ReadImage(pred_path[0])
    img_pred_data = sitk.GetArrayFromImage(img_pred)

    seg_compare = np.zeros((64, 64, 64))
    for i in range(64):
        gt_slice = img_gt_data[i, :, :]
        pred_slice = img_pred_data[i, :, :]
        # print(f'gt slice = {np.shape(gt_slice)}, seg_compare[i, :, :] = {np.shape(seg_compare[i, :, :])}')

        # True Positive
        seg_compare[i, :, :] = np.where(gt_slice == pred_slice, 40, seg_compare[i, :, :])
        # False Positive
        seg_compare[i, :, :] = np.where((gt_slice == 0) & (pred_slice != 0), 53, seg_compare[i, :, :])
        # False Negative
        seg_compare[i, :, :] = np.where((gt_slice != 0) & (pred_slice == 0), 70, seg_compare[i, :, :])
        # True Negative
        seg_compare[i, :, :] = np.where((gt_slice == 0) & (pred_slice == 0), 0, seg_compare[i, :, :])

    seg_img = sitk.GetImageFromArray(seg_compare[:, :, :])
    os.chdir(f)
    sitk.WriteImage(seg_img[:, :, :], file_name)
    print(f'{file_name} saved in {os.getcwd()}')
    if idx == 59:
        break