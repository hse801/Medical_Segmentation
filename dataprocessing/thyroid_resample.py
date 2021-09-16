import os
import nibabel as nb
import numpy as np
import SimpleITK as sitk
import glob
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

'''
rescale_slope = 1, rescale_intercept = -1024
window_center = 40, window_width = 400
'''

file_path = glob.glob('D:/0902_Thyroid/ThyroidSPECT Dataset/*/*/')

for f in file_path:
    # print(f'f = {f}')
    ct_list = glob.glob(f + 'CTCT StandardThyroid SPECT/*.DCM')
    mask_list = glob.glob(f + 'Q.Metrix Organs/*.DCM')
    spect_list = glob.glob(f + 'Q.Metrix_Transaxials_IsotropNM/*.DCM')

    if len(ct_list) == 0:
        raise Exception('No CT files')

    if len(mask_list) != 1:
        raise Exception('Mask file # is not 1')

    if len(spect_list) != 1:
        raise Exception('Spect file # is not 1')

    ct_img = pydicom.read_file(ct_list[0])
    ct_arr = ct_img.pixel_array.astype(np.float32)
    rescale_slope = ct_img.RescaleSlope
    rescale_intercept = ct_img.RescaleIntercept
    window_center = ct_img.WindowCenter
    window_width = ct_img.WindowWidth

    ct_spacing = ct_img.PixelSpacing
    ct_spacing_btw_slice = ct_img.SpacingBetweenSlices
    ct_position = ct_img.ImagePositionPatient
    ct_orientation = ct_img.ImageOrientationPatient
    ct_slice_location = ct_img.SliceLocation
    ct_slice_thickness = ct_img.SliceThickness
    # '0020', '0032' = ImagePositionPatient
    # ct_slice_num = ct_img.NumberOfSlices
    print(f'ct len = {len(ct_list)}, ct_spacing = {ct_spacing}, ct_position = {ct_position}, ct_orientation = {ct_orientation}')
    print(f'ct arr size = {ct_arr.shape}, ct_spacing_btw_slice = {ct_spacing_btw_slice}, ct_slice_location = {ct_slice_location}, ct_slice_thickness = {ct_slice_thickness}')

    spect_img = pydicom.read_file(spect_list[0])
    spect_arr = spect_img.pixel_array.astype(np.float32)
    spect_slice_num = spect_img.NumberOfSlices
    print(f'spect_arr size = {spect_arr.shape}, spect_slice_num = {spect_slice_num}')

    mask_img = pydicom.read_file(mask_list[0])
    mask_arr = mask_img.pixel_array.astype(np.float32)
    mask_slice_num = mask_img.NumberOfSlices
    print(f'mask_arr size = {mask_arr.shape}, mask_slice_num = {mask_slice_num}')

    # full_ct = np.zeros((ct_arr.shape[0], ct_arr.shape[1], len(ct_list)))
    # slice_position_tmp = np.zeros((len(ct_list), 1))
    # print(f'full ct = {full_ct.shape}, slice position = {slice_position_tmp.shape}')
    # for i, ct in enumerate(ct_list):
    #     # print(f'i = {i}, ct = {ct}')
    #     ct_slice_dcm = pydicom.read_file(ct)
    #     ct_slice_arr = ct_slice_dcm.pixel_array.astype(np.float32)
    #     full_ct[:, :, i] = ct_slice_arr
    #     slice_position_tmp[i] = ct_slice_dcm.ImagePositionPatient[2]
    #     # print(f'ct_slice_arr = {ct_slice_arr.shape}, slice_position[i] = {slice_position[i]}')
    #
    # # print(f'slice_position_tmp = {slice_position_tmp}')
    #
    # if ct_spacing_btw_slice > 0:
    #     if_reverse = False # ascend
    #     print(f'ascend')
    # else:
    #     if_reverse = True # descend
    #     print(f'descend')
    #
    # slice_position = sorted(slice_position_tmp, reverse=if_reverse)
    # # idx_list_slice = [slice_position_tmp.index(x) for x in sorted(slice_position)]
    # idx_list_slice = sorted(range(len(slice_position_tmp)), key=lambda k: slice_position_tmp[k])
    # print(f'slice_position = {slice_position}')
    # print(f'idx_list_slice = {idx_list_slice}')
    break


def convert_img(input_name, output_name, new_width=None):
    image_file_reader = sitk.ImageFileReader()
    # only read DICOM images
    image_file_reader.SetImageIO('GDCMImageIO')
    image_file_reader.SetFileName(input_name)
    image_file_reader.ReadImageInformation()
    image_size = list(image_file_reader.GetSize())
    if len(image_size) == 3 and image_size[2] == 1:
        image_size[2] = 0
    # print(f'image_size = {image_size}') # image_size = [512, 512, 1]
    image_file_reader.SetExtractSize(image_size)
    image = image_file_reader.Execute()
    sitk.Resample()


# for f in file_path:
#     ct_list = glob.glob(f + 'CTCT StandardThyroid SPECT/*.DCM')
#     input_name = ct_list[0].split('\\')[-1]
#     # print(f'input_name = {input_name}')
#     output_name = 'ct_resmpl.nii.gz'
#     convert_img(input_name=ct_list[0], output_name=output_name)

# for f in file_path:
#     print(f'f = {f}')
#     ct_list = glob.glob(f + 'CTCT StandardThyroid SPECT/*.DCM')
#     mask_list = glob.glob(f + 'Q.Metrix Organs/*.DCM')
#     spect_list = glob.glob(f + 'Q.Metrix_Transaxials_IsotropNM/*.DCM')
#     print(f'ct len = {len(ct_list)}, mask len = {len(mask_list)}, spect len = {len(spect_list)}')
#
#     ct_img = pydicom.read_file(ct_list[0])
#     ct_arr = ct_img.pixel_array.astype(np.float32)
#     rescale_slope = ct_img.RescaleSlope
#     rescale_intercept = ct_img.RescaleIntercept
#     window_center = ct_img.WindowCenter
#     window_width = ct_img.WindowWidth
#     # print(f'rescale_slope = {rescale_slope}, rescale_intercept = {rescale_intercept}')
#     # print(f'window_center = {window_center}, window_width = {window_width}')
#
#     # ct_arr = ct_arr.astype(np.float32)
#     ct_arr = (ct_arr / (2 ** ct_img.BitsStored))
#     print(f'ct_img.BitsStored = {ct_img.BitsStored}, ct_arr = {ct_arr.shape}')
#     ct_arr = ct_arr * rescale_slope + rescale_intercept




# ct_img = pydicom.read_file(ct_list[0])
# ct_arr = ct_img.pixel_array
# print(f'ct arr shape = {ct_arr.shape}')
# print(ct_img)


def dcm2niis():
    ct_list = glob.glob('D:/0902_Thyroid/ThyroidSPECT Dataset/10414125/*/CTCT StandardThyroid SPECT/IM1.DCM')
    print(f'file list len = {len(ct_list)}')
    os.chdir('C:/Users/Bohye/Desktop/mricron')
    for i in ct_list:
        print(f'file list: {i}')
        os.system('dcm2nii.exe ' + i)
        # break
    print(f'file list len = {len(ct_list)}')
# dcm2niis()



