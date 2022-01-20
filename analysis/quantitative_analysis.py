import os
import nibabel as nb
import numpy as np
import tables
import SimpleITK as sitk
import glob
import torch
import pydicom
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interpn
import pandas as pd


def dcm_header(file_path):
    dose_info = pydicom.dcmread(file_path[0])

    sensitivity = dose_info[0x0033,0x1170][0][0x0033,0x101e][732]
    print(f'sensitivity = {sensitivity}')

    ScanM = int(dose_info[0x0008, 0x0022][4:6])  # Month
    ScanD = int(dose_info[0x0008, 0x0022][6:8])  # Date
    ScanT = int(dose_info[0x0008, 0x0032][0:2])  # Hours
    Scanm = int(dose_info[0x0008, 0x0032][2:4])  # min
    ScanS = int(dose_info[0x0008, 0x0032][4:6])  # sec
    Scan_time = ScanM * 30 * 24 * 3600 + ScanD * 24 * 3600 + ScanT * 3600 + Scanm * 60 + ScanS

    halfT = dose_info[0x0033, 0x1170][0][0x0033, 0x101e][15]  # seconds unit
    PreAct = dose_info[0x0033, 0x1170][0][0x0033, 0x101e][5]  # mCi
    PostAct = dose_info[0x0033, 0x1170][0][0x0033, 0x101e][6]  # mCi

    preM = int(str(dose_info[0x0033, 0x1170][0][0x0033, 0x101e][2])[4:6])  # Month
    preD = int(str(dose_info[0x0033, 0x1170][0][0x0033, 0x101e][2])[6:8])  # Date
    preT = int(str(dose_info[0x0033, 0x1170][0][0x0033, 0x101e][2])[8:10])  # Hour
    prem = int(str(dose_info[0x0033, 0x1170][0][0x0033, 0x101e][2])[10:12])  # min
    preS = int(str(dose_info[0x0033, 0x1170][0][0x0033, 0x101e][2])[12:14])  # sec
    pre_time = preM * 30 * 24 * 3600 + preD * 24 * 3600 + preT * 3600 + prem * 60 + preS

    injM = int(str(dose_info[0x0033, 0x1170][0][0x0033, 0x101e][20])[4:6])  # Month
    injD = int(str(dose_info[0x0033, 0x1170][0][0x0033, 0x101e][20])[6:8])  # Date
    injT = int(str(dose_info[0x0033, 0x1170][0][0x0033, 0x101e][20])[8:10])  # Hour
    injm = int(str(dose_info[0x0033, 0x1170][0][0x0033, 0x101e][20])[10:12])  # min
    injS = int(str(dose_info[0x0033, 0x1170][0][0x0033, 0x101e][20])[12:14])  # sec
    inj_time = injM * 30 * 24 * 3600 + injD * 24 * 3600 + injT * 3600 + injm * 60 + injS

    postM = int(str(dose_info[0x0033, 0x1170][0][0x0033, 0x101e][3])[4:6])  # Month
    postD = int(str(dose_info[0x0033, 0x1170][0][0x0033, 0x101e][3])[6:8])  # Date
    postT = int(str(dose_info[0x0033, 0x1170][0][0x0033, 0x101e][3])[8:10])  # Hour
    postm = int(str(dose_info[0x0033, 0x1170][0][0x0033, 0x101e][3])[10:12])  # min
    postS = int(str(dose_info[0x0033, 0x1170][0][0x0033, 0x101e][3])[12:14])  # sec
    post_time = postM * 30 * 24 * 3600 + postD * 24 * 3600 + postT * 3600 + postm * 60 + postS

    PreAct_DC = PreAct * np.exp(math.log(2) * (pre_time - inj_time) / halfT)
    PostAct_DC = PostAct * np.exp(math.log(2) * (post_time - inj_time) / halfT)
    inj_dose = 1000 * (PreAct_DC - PostAct_DC)  # uCi unit
    decay_factor = np.exp(math.log(2) * (Scan_time - inj_time) / halfT)
    print(f'Scan_time = {Scan_time},pre_time = {pre_time},inj_time = {inj_time},post_time = {post_time}')
    print(f'PreAct_DC = {PreAct_DC}, PostAct_DC = {PostAct_DC}, inj_dose = {inj_dose}, decay_factor = {decay_factor}')
    return sensitivity, decay_factor, inj_dose
    # break


def read_img(file_path):
    target_img = nb.load(file_path[0])
    target_img_data = target_img.get_fdata()
    target_affine = target_img.affine
    print(f'affine = {target_affine}')


def compare_affine(src_file, dst_file):
    print(f'-------------------------------')
    src_img = nb.load(src_file[0])
    src_img_data = src_img.get_fdata()
    src_affine = src_img.affine
    print(f'src_affine = {src_affine}, shape = {np.shape(src_affine)}, min = {np.min(src_img_data)}, max = {np.max(src_img_data)}')

    dst_img = nb.load(dst_file[0])
    dst_img_data = dst_img.get_fdata()
    dst_affine = dst_img.affine
    print(f'dst_affine = {dst_affine}, shape = {np.shape(dst_affine)}, max = {np.max(dst_img_data)}')

    if not np.array_equal(src_affine, dst_affine):
        print(f'affine diff')
        print(f'src_file = {src_file[0]}')
        # dst_affine = src_affine
        # return


def compare_affine_sitk(src_file, dst_file):
    dst_img = sitk.ReadImage(dst_file[0])
    dst_size = dst_img.GetSize()
    dst_spacing = dst_img.GetSpacing()
    dst_origin = dst_img.GetOrigin()
    print(f'dst_size = {dst_size}, dst_spacing = {dst_spacing}, dst_origin = {dst_origin}')

    src_img = sitk.ReadImage(src_file[0])
    src_size = np.array(src_img.GetSize())
    src_spacing = np.array(src_img.GetSpacing())
    src_origin = np.array(src_img.GetOrigin())
    print(f'src_size = {src_size}, src_spacing = {src_spacing}, src_origin = {src_origin}')



def get_img(spect_path, mask_path):
    """
    calculate spect count of the mask region

    :param spect_path:
    :param mask_path:
    :return:
    """
    # print(f'-------------------------------')
    spect_img = nb.load(spect_path[0])
    spect_img_data = spect_img.get_fdata()
    spect_affine = spect_img.affine
    # print(f'spect_affine = {spect_affine}, shape = {np.shape(spect_affine)}, min = {np.min(spect_img_data)}, max = {np.max(spect_img_data)}')

    mask_img = nb.load(mask_path[0])
    mask_img_data = mask_img.get_fdata()
    mask_affine = mask_img.affine
    # print(f'mask_affine = {mask_affine}, shape = {np.shape(mask_affine)}, max = {np.max(mask_img_data)}')

    if not np.array_equal(spect_affine, mask_affine):
        print(f'affine diff')
        print(f'spect path = {spect_path[0]}')
        mask_affine = spect_affine
        # return
    mask_img_data = np.where(mask_img_data > 0, 1, 0)
    region_data = spect_img_data * mask_img_data
    region_sum = np.sum(region_data)

    total_sum = np.sum(spect_img_data)
    print(f'region_data shape = {np.shape(region_data)}, region_sum = {region_sum}, total_sum = {total_sum}, ratio = {region_sum / total_sum}')
    return region_sum


# def get_id(decay_factor, sensitivity, inj_dose, img_val):
#     percent_id = img_val * decay_factor / sensitivity / inj_dose
#     return percent_id

def resample_volume(src_file, dst_file, save_dir):
    """
    resample src file to have same voxel size and origin with dst file

    :param src_file:
    :param dst_file:
    :return:
    """
    # src_img = nb.load(src_file[0])
    # src_img_data = src_img.get_fdata()
    # src_affine = src_img.affine

    src_img = sitk.ReadImage(src_file[0])
    dst_img = sitk.ReadImage(dst_file[0])

    file_name = src_file[0].split('\\')[-1].split('.')[0] + '_rsmpl.nii.gz'
    # print(f'file_name = {file_name}')

    resample = sitk.ResampleImageFilter()

    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    resample.SetOutputDirection(src_img.GetDirection())
    resample.SetDefaultPixelValue(0)

    # set output spacing
    dst_size = dst_img.GetSize()
    dst_spacing = dst_img.GetSpacing()
    dst_origin = dst_img.GetOrigin()
    resample.SetOutputSpacing(dst_spacing)
    print(f'dst_size = {dst_size}, dst_spacing = {dst_spacing}, dst_origin = {dst_origin}')

    # set output size and origin
    src_size = np.array(src_img.GetSize())
    src_spacing = np.array(src_img.GetSpacing())
    src_origin = np.array(src_img.GetOrigin())
    print(f'src_size = {src_size}, src_spacing = {src_spacing}, src_origin = {src_origin}')

    dst_size_no_shift = np.int16(np.ceil(src_size * src_spacing / dst_spacing))
    shift_amount = np.int16(np.floor((dst_size_no_shift - dst_size)/2)) * dst_spacing

    resample.SetSize(dst_size)
    resample.SetOutputOrigin(dst_origin)
    resample.SetOutputSpacing(dst_spacing)

    resampled_src = resample.Execute(src_img)
    resampled_size = np.array(resampled_src.GetSize())
    resampled_spacing = np.array(resampled_src.GetSpacing())
    resampled_origin = np.array(resampled_src.GetOrigin())
    print(f'resampled_size = {resampled_size}, resampled_spacing = {resampled_spacing}, resampled_origin = {resampled_origin}\n')

    os.chdir(save_dir)
    # sitk.WriteImage(resampled_src, file_name)
    print(f'{file_name} saved in {save_dir}\n\n')


def resample_nb(src_file, dst_file, save_dir):
    """
    resample src file to have same voxel size and origin with dst file
    --> 이게 최종
    :param src_file:
    :param dst_file:
    :param save_dir:
    :return:
    """
    file_name = src_file[0].split('\\')[-1].split('.')[0] + '_rsmpl.nii.gz'
    # file_name = 'crop_mask.nii.gz'

    src_img = nb.load(src_file[0])
    src_img_data = src_img.get_fdata()
    src_affine = src_img.affine
    print(f'src_affine = {src_affine}, shape = {np.shape(src_affine)}, min = {np.min(src_img_data)}, max = {np.max(src_img_data)}')

    dst_img = nb.load(dst_file[0])
    dst_img_data = dst_img.get_fdata()
    dst_affine = dst_img.affine
    print(f'dst_affine = {dst_affine}, shape = {np.shape(dst_affine)}, max = {np.max(dst_img_data)}')

    os.chdir(save_dir)
    nb.save(nb.Nifti1Pair(src_img_data, dst_affine), file_name)
    print(f'{file_name} saved in {save_dir}\n\n')


path_list = glob.glob('D:/0902_Thyroid/ThyroidSPECT Dataset/*/Tc Thyroid SPECT/')
# path_list = glob.glob('D:/0902_Thyroid/29085932/Tc Thyroid SPECT/')

id_list = []

for p in path_list:
    spect_dcm_file = glob.glob(p + 'Q.Metrix_Transaxials_IsotropNM/IM*.DCM')
    mask_dcm_file = glob.glob(p + 'Q.Metrix Organs/IM*.DCM')

    mask_nii_file = glob.glob(p + 'crop_mask.nii.gz')
    spect_nii_file = glob.glob(p + 'crop_spect.nii.gz')

    unet_file = glob.glob(p + '3DUNET*rsmpl.nii.gz')
    attention_file = glob.glob(p + 'ATTENTION*rsmpl.nii.gz')
    proposed_file = glob.glob(p + 'Proposed*rsmpl.nii.gz')

    # unet_file = glob.glob(p + '3DUNET_0.nii.gz')
    # attention_file = glob.glob(p + 'ATTENTION*_0.nii.gz')
    # rsmpl_proposed = glob.glob(p + 'Proposed_0.nii.gz')

    # Resample the proposed prediction file as mask file for quantification
    # resample_nb(src_file=rsmpl_proposed, dst_file=mask_nii_file, save_dir = p)
    # break

    # to check the affine after resampling
    # compare_affine(src_file=rsmpl_proposed, dst_file=mask_nii_file)
    # compare_affine_sitk(src_file=rsmpl_proposed, dst_file=mask_nii_file)

    # To read tag of the dicom file to get the necessary information from spect file
    # dcm_header(file_path=mask_dcm_file)
    print(f'spect_dcm_file = {spect_dcm_file}, mask_dcm_file = {mask_dcm_file}, mask_nii_file = {mask_nii_file}, spect_nii_file = {spect_nii_file}')
    print(f'unet_file = {unet_file}, attention_file = {attention_file}, proposed_file = {proposed_file}')
    sensitivity, decay_factor, inj_dose = dcm_header(file_path=mask_dcm_file)

    # calculate spect count of the mask region
    # read_img(file_path=mask_nii_file)
    mask_sum = get_img(spect_path=spect_nii_file, mask_path=mask_nii_file)
    unet_sum = get_img(spect_path=spect_nii_file, mask_path=unet_file)
    attention_sum = get_img(spect_path=spect_nii_file, mask_path=attention_file)
    proposed_sum = get_img(spect_path=spect_nii_file, mask_path=proposed_file)

    # region_activity = region_sum * decay_factor / sensitivity / (60 * 20)
    # divide by 2 : duration time 2 minutes because of two gamma camera
    mask_percent_id = mask_sum * decay_factor / sensitivity / inj_dose * 100 / 2
    unet_percent_id = unet_sum * decay_factor / sensitivity / inj_dose * 100 / 2
    attention_percent_id = attention_sum * decay_factor / sensitivity / inj_dose * 100 / 2
    proposed_percent_id = proposed_sum * decay_factor / sensitivity / inj_dose * 100 / 2

    print(f'mask_percent_id = {mask_percent_id}, mask_sum = {mask_sum}')
    print(f'unet_percent_id = {unet_percent_id}, unet_sum = {unet_sum}')
    print(f'attention_percent_id = {attention_percent_id}, attention_sum = {attention_sum}')
    print(f'proposed_percent_id = {proposed_percent_id}, proposed_sum = {proposed_sum}')

    percent_ids = {}
    percent_ids.update({'mask': mask_percent_id,
                         'unet': unet_percent_id,
                         'attention': attention_percent_id,
                         'proposed': proposed_percent_id})
    id_list.append(percent_ids)

    print('---------------------------------------------------')

print(f'Percent ID dataframe: ')
eval_df = pd.DataFrame(id_list, columns=['mask', 'unet', 'attention', 'proposed'])
# Add row of Mean value of each metrics
eval_df.loc['Mean'] = eval_df.mean()
eval_df.loc['Median'] = eval_df.median()
eval_df.loc['Std'] = eval_df.std()
print(eval_df)
print(eval_df.loc['Mean'])
# os.chdir(PATH + save_folder)
eval_df.to_csv('%ID_analysis_0120.csv', mode='w')
print(f'ID_analysis csv saved in {os.getcwd()}')
    # resample_volume(src_file=proposed_file, dst_file=mask_nii_file, save_dir = p)
    # break
