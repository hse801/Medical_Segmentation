import os
import nibabel as nb
import numpy as np
import tables
import SimpleITK as sitk
import glob
import torch
import lib.medloaders as dataloaders
import lib.medzoo as medzoo
from lib.losses3D import DiceLoss
from lib.visual3D_temp.BaseWriter import TensorboardWriter

"""
Crop the 0902_Thyroid data
Before: 256 x 256 x 256
After: 64 x 64 x 64

Using prediction result of trained model 
--> Have to work on!!!!!!!

"""


def predictor(PATH, data_loader):

    # model_path = PATH + 'UNET3D_29_06___17_24_thyroid_/UNET3D_29_06___17_24_thyroid__BEST.pth'
    model_path = PATH + 'UNET3D_11_06___07_07_thyroid_/UNET3D_11_06___07_07_thyroid__BEST.pth'

    # path_list = glob.glob('D:/0902_Thyroid/ThyroidSPECT Dataset/*/Tc Thyroid SPECT/')
    path_list = glob.glob('E:/HSE/Thyroid/Dicom/*/')

    # ex_path = 'E:/HSE/Thyroid/Dicom/1.2.410.2000010.82.2291.1002869190726010/CT_rsmpl.nii.gz'
    # ex_img = nb.load(ex_path)

    model = medzoo.UNet3D(in_channels=1, n_classes=1, base_n_filter=12)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # model = model.cuda()
    model.eval()

    for batch_idx, input_tuple in enumerate(data_loader):
        with torch.no_grad():
            print(f'os.getcwd = {os.getcwd()}')
            input_tensor, target = input_tuple
            input_tensor, target = input_tensor.cpu(), target.cpu()
            input_tensor.requires_grad = False

            output = model(input_tensor)
            criterion = DiceLoss(classes=1)
            loss, per_ch_score = criterion(output, target)
            print(f'loss = {loss}, per_ch_score = {per_ch_score}')

            output = output.squeeze()
            output_arr = output.cpu().numpy()
            # print(f'output_arr type = {type(output_arr)}, output_arr size = {np.shape(output_arr)}')
            # print(f'output_arr min = {np.min(output_arr)}, output_arr max = {np.max(output_arr)}')
            # file_name = f'pred_11_06_07_07_{batch_idx}.nii.gz'

            # set threshold to the predicted image
            output_arr = np.where(output_arr > 0, 1, 0)
            nzero = output_arr.nonzero()
            x_mid = int((min(nzero[2]) + max(nzero[2])) / 2)
            y_mid = int((min(nzero[1]) + max(nzero[1])) / 2)
            z_mid = int((min(nzero[0]) + max(nzero[0])) / 2)

            # crop_file(i, batch_idx)

            # output_img = nb.Nifti1Image(output_arr, ex_img.affine, ex_img.header)
            # # print(f'output_img type = {type(output_img)}, output_img size = {output_img.size()}')
            # os.chdir(path_list[batch_idx])
            # nb.save(output_img, file_name)
            # print(f'{file_name} saved in {os.getcwd()}')
            print(f'prediction done -------------------------------\n')
        # break


def crop_file(folder_path, count):
    ct_file = folder_path + 'CT_rsmpl.nii.gz'
    mask_file = folder_path + 'Mask_rsmpl.nii.gz'
    spect_file = folder_path + 'SPECT.nii.gz'
    pred_file = glob.glob(folder_path + f'pred_{count}.nii.gz')
    print(f'pred_file = {pred_file}')

    print(f'ct path = {ct_file}')
    img_pred = nb.load(pred_file[0])
    img_pred_data = img_pred.get_fdata()
    nzero = img_pred_data.nonzero()
    # print(f'nzero shape = {np.shape(nzero)}')
    # print(f'nzero = {nzero}')
    # print(f'nzero[0] = {min(nzero[0])}')
    # print(f'nzero[2] = {nzero}')
    x_mid = int((min(nzero[2]) + max(nzero[2])) / 2)
    y_mid = int((min(nzero[1]) + max(nzero[1])) / 2)
    z_mid = int((min(nzero[0]) + max(nzero[0])) / 2)
    print(f'xmid = {x_mid}, ymid = {y_mid}, zmid = {z_mid}')

    os.chdir(folder_path)

    cropped_ct_img = crop_file_to_img(x_mid, y_mid, z_mid, ct_file)
    nb.save(cropped_ct_img, f'crop_ct_{count}.nii.gz')

    cropped_spect_img = crop_file_to_img(x_mid, y_mid, z_mid, spect_file)
    nb.save(cropped_spect_img, f'crop_spect_{count}.nii.gz')

    cropped_mask_img = crop_file_to_img(x_mid, y_mid, z_mid, mask_file)
    nb.save(cropped_mask_img, f'crop_mask_{count}.nii.gz')

    print(f'files saved in {os.getcwd()}')
    # x_max =


def crop_file_to_img(x_mid, y_mid, z_mid, file_to_crop):

    file_img = nb.load(file_to_crop)
    file_arr = file_img.get_fdata()

    # set the same voxel size with file before crop
    # file_spacing = file_img.GetSpacing()

    # file_origin = file_img.GetOrigin()
    # file_direction = file_img.GetDirection()
    # print(f'file_spacing = {file_spacing}, file_origin = {file_origin}, file_direction = {file_direction}')
    z_start, z_end = check_in_range(z_mid, crop_range=32, file_dim=128)
    y_start, y_end = check_in_range(y_mid, crop_range=32, file_dim=128)
    x_start, x_end = check_in_range(x_mid, crop_range=32, file_dim=128)
    cropped_arr = file_arr[z_start:z_end, y_start:y_end, x_start:x_end]
    # print(f'file_img.affine = {file_img.affine}')
    cropped_img = nb.Nifti1Image(cropped_arr, file_img.affine, file_img.header)
    # cropped_img.SetSpacing(file_spacing)
    # cropped_img.CopyInformation(file_img)
    # crop_spacing = cropped_img.GetSpacing()
    # crop_origin = cropped_img.GetOrigin()
    # crop_direction = cropped_img.GetDirection()
    # print(f'crop_spacing = {crop_spacing}, crop_origin = {crop_origin}, crop_direction = {crop_direction}')

    return cropped_img


def check_in_range(mid, crop_range, file_dim):
    if mid + crop_range > file_dim:
        start = file_dim - crop_range * 2
        end = file_dim
        print('range over max')
    elif mid - crop_range < 0:
        start = 0
        end = crop_range * 2
        print('range under min')
    else:
        start = mid - crop_range
        end = mid + crop_range

    return start, end


_, _, pred_loader = dataloaders.thyroid_dataloader.generate_thyroid_dataset()
PATH = 'E:/HSE/Medical_Segmentation/saved_models/UNET3D_checkpoints/'
# model_path = PATH + 'UNET3D_29_06___17_24_thyroid_/UNET3D_29_06___17_24_thyroid__BEST.pth'
# model_path = PATH + 'UNET3D_29_06___17_24_thyroid_/UNET3D_29_06___17_24_thyroid__last_epoch.pth'

predictor(PATH=PATH, data_loader=pred_loader)

folder_path = glob.glob('D:/0902_Thyroid/ThyroidSPECT Dataset/*/Tc Thyroid SPECT/')
# print(f'type = {type(folder_path)}, len = {len(folder_path)}')
count = 0
# for i in folder_path:
#     crop_file(i, count)
#     count += 1
#     print(f'count = {count}')
#     print('-------------------------------\n')
#     break
# ct_path = glob.glob('E:/HSE/Thyroid/Dicom/*/CT_rsmpl.nii.gz')
# mask_path = glob.glob('E:/HSE/Thyroid/Dicom/*/Mask_rsmpl.nii.gz')