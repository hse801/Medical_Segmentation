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


def predictor(PATH, data_loader):

    # model_path = PATH + 'UNET3D_29_06___17_24_thyroid_/UNET3D_29_06___17_24_thyroid__BEST.pth'
    # 2 channel label, 2 classes
    model_path = PATH + 'UNET3D_22_44___07_12_thyroid_/UNET3D_22_44___07_12_thyroid__BEST.pth'
    path_list = glob.glob('E:/HSE/Thyroid/Dicom/*/')

    model = medzoo.UNet3D(in_channels=1, n_classes=2, base_n_filter=24)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # model = model.cuda()
    model.eval()

    for batch_idx, input_tuple in enumerate(data_loader):
        with torch.no_grad():
            # input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
            input_tensor, target = input_tuple
            # print(f'input tuple len = {len(input_tuple)}, input tuple = {input_tuple}')
            # print(f'target type = {type(target)}, target = {target.nonzero()}')
            # input_tensor, target = input_tensor.cuda(), target.cuda()
            input_tensor, target = input_tensor.cpu(), target.cpu()
            input_tensor.requires_grad = False

            output = model(input_tensor)
            criterion = DiceLoss(classes=2)
            print(f'output size = {output.size()}, target = {target.size()}')
            loss, per_ch_score = criterion(output, target)
            print(f'loss = {loss}, per_ch_score = {per_ch_score}')

            # print(f'bf output type = {output.type()}, output size = {output.size()}, ')
            output = output.squeeze()
            # loss, per_ch_score = criterion(output, target)

            # print(f'af output size = {output.size()}')
            output_arr = output.cpu().numpy()
            print(f'output_arr type = {type(output_arr)}, output_arr size = {np.shape(output_arr)}')
            print(f'output_arr min = {np.min(output_arr)}, output_arr max = {np.max(output_arr)}')

            # file_name1 = f'pred_2ch_1_{batch_idx}.nii.gz'
            # file_name2 = f'pred_2ch_2_{batch_idx}.nii.gz'
            file_name = f'pred_2ch_{batch_idx}.nii.gz'

            # os.chdir('E:/HSE/Medical_Segmentation/saved_models/UNET3D_checkpoints/UNET3D_17_08___07_06_thyroid_/prediction/')
            # os.mkdir('prediction/')
            # print(f'current: {os.getcwd()}')
            # os.mkdir('prediction/')
            # output_arr = np.where(output_arr > 0, 1, (np.where(output_arr < -3, 0, 1)))

            # set threshold to the predicted image
            output_arr = np.where(output_arr > 0, 1, 0)
            # create combined array of left and right labels
            output_combined_arr = output_arr[0, :, :, :] + output_arr[1, :, :, :]

            # output_img_1 = sitk.GetImageFromArray(output_arr[0, :, :, :])
            # output_img_2 = sitk.GetImageFromArray(output_arr[1, :, :, :])
            # output_combined = output_img_1 + output_img_2
            # output_combined_arr = sitk.GetArrayFromImage(output_combined)

            # if value is 2, change to 1
            output_combined_arr = np.where(output_combined_arr > 1, 1, output_combined_arr)
            # print(f'output_combined_arr max = {output_combined_arr.max()}')
            output_combined = sitk.GetImageFromArray(output_combined_arr)
            # print(f'output_img type = {type(output_img)}, output_img size = {output_img.size()}')
            os.chdir(path_list[batch_idx])
            # sitk.WriteImage(output_img_1[:, :, :], file_name1)
            # sitk.WriteImage(output_img_2[:, :, :], file_name2)
            sitk.WriteImage(output_combined[:, :, :], file_name)
            print(f'{file_name} saved in {os.getcwd()}')
            print(f'prediction done -------------------------------\n')
            # print(f'output type = {output.type()}, output size = {output.size()}')
        # break


_, _, pred_loader = dataloaders.thyroid_dataloader.generate_thyroid_dataset()
PATH = 'E:/HSE/Medical_Segmentation/saved_models/UNET3D_checkpoints/'
# model_path = PATH + 'UNET3D_29_06___17_24_thyroid_/UNET3D_29_06___17_24_thyroid__BEST.pth'
# model_path = PATH + 'UNET3D_29_06___17_24_thyroid_/UNET3D_29_06___17_24_thyroid__last_epoch.pth'

predictor(PATH=PATH, data_loader=pred_loader)
