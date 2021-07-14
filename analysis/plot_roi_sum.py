import numpy as np
import glob
import SimpleITK as sitk
import nibabel as nb
from matplotlib import pyplot as plt

# def get_sum_arr(file_path, file_name):

def draw_histogram(file_path):
    mask_path = file_path + 'crop_mask.nii.gz'
    model1_path = glob.glob(file_path + 'pred_22_55_07_11*.nii.gz')
    model2_path = glob.glob(file_path + 'pred_2ch_combined*.nii.gz')
    model3_path = glob.glob(file_path + 'pred_15_03*.nii.gz')
    print(f'mask_path = {mask_path}')

    img_mask = sitk.ReadImage(mask_path)
    img_mask_data = sitk.GetArrayFromImage(img_mask)
    img_mask_data[img_mask_data > 0] = 1

    print(f'img_mask_data = {img_mask_data.shape}')
    mask_sum = np.sum(img_mask_data, axis=0)
    print(f'1 mask_sum shape = {mask_sum.shape}')
    mask_sum = np.sum(mask_sum, axis=0)
    print(f'2 mask_sum shape = {mask_sum.shape}')
    print(f'mask_sum type = {type(mask_sum)}')

    img_md1 = sitk.ReadImage(model1_path[0])
    img_md1_data = sitk.GetArrayFromImage(img_md1)
    md1_sum = np.sum(img_md1_data, axis=0)
    md1_sum = np.sum(md1_sum, axis=0)

    img_md2 = sitk.ReadImage(model2_path[0])
    img_md2_data = sitk.GetArrayFromImage(img_md2)
    md2_sum = np.sum(img_md2_data, axis=0)
    md2_sum = np.sum(md2_sum, axis=0)

    img_md3 = sitk.ReadImage(model3_path[0])
    img_md3_data = sitk.GetArrayFromImage(img_md3)
    md3_sum = np.sum(img_md3_data, axis=0)
    md3_sum = np.sum(md3_sum, axis=0)

    return mask_sum, md1_sum, md2_sum, md3_sum


path_list = glob.glob('E:/HSE/Thyroid/Dicom/*/')
count, mask_total, md1_total, md2_total, md3_total = 0, 0, 0, 0, 0

for i in path_list:
    mask, md1, md2, md3 = draw_histogram(i)
    mask_total += mask
    md1_total += md1
    md2_total += md2
    md3_total += md3
    x_range = np.arange(64)
    count += 1
    print(f'count = {count}')
    if count > 60:
        print(f'md1_total = {md1_total}, md2_total = {md2_total}, md3_total = {md3_total}')
        plt.plot(x_range, mask_total, label='Ground Truth')
        plt.plot(x_range, md1_total, label='Model 1')
        plt.plot(x_range, md2_total, label='Model 2')
        plt.plot(x_range, md3_total, label='Model 3')
        plt.legend(loc='best')

        plt.show()
        break
