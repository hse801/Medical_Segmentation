import os
import nibabel as nb
import numpy as np
import SimpleITK as sitk
import glob
import pydicom
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import nibabel
from scipy.interpolate import interpn
from typing import List
import cv2, pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut


"""
preprocessing for ct attenuation map
test for segmentation with ct-free generated attenuation map

label.nii.gz
prediction.nii.gz
mask.nii.gz

convert dcm files to nii.gz format
resample mask to match size with others
flip label and prediction
convert attenuation map to ct scale 
resize the data
test with segmentation model trained with ct images

"""


def view_dcm(src_path):
    window_center = -600
    window_width = 1600

    # CT image
    # dicom_path = './1-29.dcm'
    slice = pydicom.read_file(src_path)
    s = int(slice.RescaleSlope)
    b = int(slice.RescaleIntercept)
    image = s * slice.pixel_array + b

    plt.subplot(1, 3, 1)
    plt.title('DICOM -> Array')
    plt.imshow(image, cmap='gray')

    # apply_modality_lut( ) & apply_voi_lut( )
    slice.WindowCenter = window_center
    slice.WindowWidth = window_width
    image = apply_modality_lut(image, slice)
    image2 = apply_voi_lut(image, slice)
    plt.subplot(1, 3, 2)
    plt.title('apply_voi_lut( )')
    plt.imshow(image2, cmap='gray')

    # normalization
    image3 = np.clip(image, window_center - (window_width / 2), window_center + (window_width / 2))
    plt.subplot(1, 3, 3)
    plt.title('normalize')
    plt.imshow(image3, cmap='gray')

    plt.show()


def convert_to_HU(src_path):
    img_src = sitk.ReadImage(src_path[0])
    data_src = sitk.GetArrayFromImage(img_src)


def reverse_along_axis(data: np.array, axis: int):
    if axis == 0:
        data = data[::-1, :, :]
    elif axis == 1:
        data = data[:, ::-1, :]
    else:
        data = data[:, :, ::-1]
    return data


def crop_img(src_file: str, dst_file: str, crop_index: List[int]):

    start_index, end_index = crop_index
    src_img = nibabel.load(src_file)

    zorigin = src_img.affine[2][3]
    zspacing = src_img.affine[2][2]

    src_img_data = src_img.get_fdata()
    src_img_data = src_img_data.astype(np.float32)

    affine = src_img.affine
    affine[2][3] = zorigin + zspacing * start_index
    crop_img_data = src_img_data[:, :, start_index: end_index]

    nibabel.save(nibabel.Nifti1Pair(crop_img_data, affine), dst_file)


def interp_img(src_file: str, dst_file: str, ref_file: str):
    print(f'nii_resize_image: src={src_file} dst={dst_file} ref={ref_file}')
    src_img = nibabel.load(src_file)
    ref_img = nibabel.load(ref_file)
    src_coord = np.array([np.arange(d) for d in src_img.shape])
    ref_coord = np.array([np.arange(d) for d in ref_img.shape])
    src_img_data = src_img.get_fdata()

    for i in range(3):
        src_coord[i] = src_img.affine[i][i] * src_coord[i] + src_img.affine[i][3]
        ref_coord[i] = ref_img.affine[i][i] * ref_coord[i] + ref_img.affine[i][3]
        if src_img.affine[i][i] < 0:
            src_coord[i] = src_coord[i][::-1]
            src_img_data = reverse_along_axis(src_img_data, i)
        if ref_img.affine[i][i] < 0:
            ref_coord[i] = ref_coord[i][::-1]

    ref_mesh = np.rollaxis(np.array(np.meshgrid(*ref_coord)), 0, 4) # [xdim][ydim][zdim][3]
    src_resize_data = interpn(src_coord, src_img.get_fdata(), ref_mesh, bounds_error=False, fill_value=-1024)

    for i in range(3):
        if ref_img.affine[i][i] < 0:
            src_resize_data = reverse_along_axis(src_resize_data, i)
    src_resize_data = src_resize_data.astype(np.float32)

    import pathlib
    if pathlib.Path(dst_file).exists():
        print(f'{dst_file} already exists. will overwrite')
    src_resize_data = src_resize_data.swapaxes(0, 1)
    src_resize_data = src_resize_data[::-1, ::-1, :]

    nibabel.save(nibabel.Nifti1Pair(src_resize_data, ref_img.affine), dst_file)


def flip(src_path: str, file_name: str):
    # img_src = sitk.ReadImage(src_path[0])
    # data_src = sitk.GetArrayFromImage(img_src)
    # print(f'shape = {np.shape(data_src)}')
    src_img = nibabel.load(src_path[0])
    data_src = src_img.get_fdata()
    data_src = data_src[:, :, ::-1]
    data_crop = data_src[32:96, 32:96, :64]
    nibabel.save(nibabel.Nifti1Pair(data_crop, src_img.affine), file_name)
    # [axial, coronal, sagittal]
    # [z, y, x]
    # plt.imshow(data_src[60, :, :], cmap=cm.gray)
    # plt.show()

    return data_src


def dcm2niis(src_path):
    os.chdir('C:/Users/Bohye/Desktop/mricron')
    for i in src_path:
        print(f'file list: {i}')
        os.system('dcm2nii.exe ' + i)
        # break


nii_path = glob.glob('C:/Users/Bohye/Desktop/Unet/0/prediction/prediction.nii.gz')
dcm_path = 'C:/Users/Bohye/Desktop/Unet/0/label/label.dcm'
dcm_path_2 = glob.glob('D:/0902_Thyroid/ThyroidSPECT Dataset/10414125/*/CTCT StandardThyroid SPECT/IM1.DCM')
# dcm2niis(src_path=nii_path)
os.chdir('C:/Users/Bohye/Desktop/Unet/0/prediction/')
flip(src_path=nii_path, file_name='prediction_crop.nii.gz')
# view_dcm(src_path=dcm_path_2[0])