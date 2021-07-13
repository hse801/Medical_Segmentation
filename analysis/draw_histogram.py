import numpy as np
import glob
import SimpleITK as sitk
import nibabel as nb
from matplotlib import pyplot as plt


def draw_histogram(file_path):
    mask_path = file_path + 'crop_mask.nii.gz'
    pred_path = glob.glob(file_path + 'pred_22_55_07_11*.nii.gz')
    print(f'mask_path = {mask_path}')
    x_range = np.arange(64)

    img_mask = sitk.ReadImage(mask_path)
    img_mask_data = sitk.GetArrayFromImage(img_mask)
    img_mask_data[img_mask_data > 0] = 1

    print(f'img_mask_data = {img_mask_data.shape}')
    img_mask_sum = np.sum(img_mask_data, axis=0)
    print(f'1 img_mask_sum shape = {img_mask_sum.shape}')
    img_mask_sum = np.sum(img_mask_sum, axis=0)
    print(f'2 img_mask_sum shape = {img_mask_sum.shape}')
    print(f'img_mask_sum type = {type(img_mask_sum)}')

    img_pred = sitk.ReadImage(pred_path[0])
    img_pred_data = sitk.GetArrayFromImage(img_pred)
    img_pred_sum = np.sum(img_pred_data, axis=0)
    img_pred_sum = np.sum(img_pred_sum, axis=0)

    # plt.subplot(1, 2, 1)
    plt.plot(x_range, img_mask_sum, label='Mask')
    plt.plot(x_range, img_pred_sum, label='Prediction')
    plt.legend(loc='best')
    # plt.subplot(1, 2, 2)
    # plt.plot(img_pred_sum)

    plt.show()

    # img_pred = sitk.ReadImage(pred_path[0])
    # img_pred_data = sitk.GetArrayFromImage(img_pred)


path_list = glob.glob('E:/HSE/Thyroid/Dicom/*/')
count = 0
for i in path_list:
    draw_histogram(i)
    count += 1
    print(f'count = {count}')
    if count > 9:
        break
