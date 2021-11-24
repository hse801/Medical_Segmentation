import numpy as np
import SimpleITK as sitk
import os
import scipy
import scipy.ndimage as ndimage
import torchio as tio
import glob
from torch.nn import functional as F
from PIL import Image
from typing import List, Tuple
from PIL import Image
import matplotlib
from matplotlib import cm
from matplotlib import pyplot as plt
import cv2

"""
thyroid data
plot image of the prediction results of the models
saved as .jpg file
TP, FP, FN, FP values are assigned as RGB code

"""
path_list = glob.glob('D:/0902_Thyroid/ThyroidSPECT Dataset/32906941/Tc Thyroid SPECT/')

# def create_pred_img()
for path in path_list:
    slice_num = 27
    # file_name = 'RESUNET.jpg'
    # pred_file = glob.glob(path + '1026_2157*')

    pred_file = glob.glob(path + '3DUNET*')
    file_name = '3DUNET.jpg'

    os.chdir(path)

    mask_file = path + 'crop_mask.nii.gz'
    # print(f'mask file = {mask_path}')

    mask_img = sitk.ReadImage(mask_file)
    img_mask_data = sitk.GetArrayFromImage(mask_img)
    img_mask_data = img_mask_data
    img_mask_data[img_mask_data > 0] = 1

    img_pred = sitk.ReadImage(pred_file[0])
    arr_pred_data = sitk.GetArrayFromImage(img_pred)

    os.chdir(path)
    TP = 1
    TN = 2
    FP = 3
    FN = 4


    # enter the RGB color code
    # have to be divided with 255
    color_map = {
        'TP': (255/255,228/255,196/255),
        'TN': (255/255,255/255,255/255),
        'FP': (255/255,69/255,0/255),
        'FN': (0/255,0/255,205/255)
    }

    # color_map = {
    #     'TP': (175/255,238/255,238/255),
    #     'TN': (255/255,255/255,255/255),
    #     'FP': (255/255,69/255,0/255),
    #     'FN': (0/255,128/255,0/255)
    # }

    # color_map = {
    #     'TP': (255/255,228/255,196/255),
    #     'TN': (255/255,255/255,255/255),
    #     'FP': (255/255,69/255,0/255),
    #     'FN': (0/255,0/255,205/255)
    # }

    # ResUNet
    compare_pred = np.zeros((64, 64, 64))
    compare_pred = np.where((img_mask_data == 1) & (arr_pred_data == 1), TP, compare_pred)  # True Positive
    compare_pred = np.where((img_mask_data == 0) & (arr_pred_data == 0), TN, compare_pred)  # True Negative
    compare_pred = np.where((img_mask_data == 0) & (arr_pred_data == 1), FP, compare_pred)  # False Positive
    compare_pred = np.where((img_mask_data == 1) & (arr_pred_data == 0), FN, compare_pred)  # False Negative

    pred_slice = compare_pred[slice_num, :, :]
    map_pred = np.ones((64, 64, 3))

    map_pred[pred_slice == 1] = color_map['TP']
    map_pred[pred_slice == 2] = color_map['TN']
    map_pred[pred_slice == 3] = color_map['FP']
    map_pred[pred_slice == 4] = color_map['FN']
    plt.axis("off")

    plt.imshow(map_pred)
    # 옆에 남는 공간 없이 저장
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.show()

    print(f'image saved in {os.getcwd()}')

