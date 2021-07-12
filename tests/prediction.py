import os

import nibabel as nib
import numpy as np
import tables
import SimpleITK as sitk
import glob
import torch
import lib.medloaders as dataloaders
import lib.medzoo as medzoo
from lib.losses3D import DiceLoss
from lib.visual3D_temp.BaseWriter import TensorboardWriter


def predictor(PATH, data_loader):
    # checkpoint = torch.load(model_path)
    # model = checkpoint['model']
    # model.load_state_dict(checkpoint['state_dict'])
    # for parameter in model.parameters():
    #     parameter.requires_grad = False


    # model.load_state_dict(torch)

    # model_path = PATH + 'UNET3D_29_06___17_24_thyroid_/UNET3D_29_06___17_24_thyroid__BEST.pth'
    model_path = PATH + 'UNET3D_20_40___07_08_thyroid_/UNET3D_20_40___07_08_thyroid__BEST.pth'
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
            criterion = DiceLoss(classes=1)
            loss, per_ch_score = criterion(output, target)
            print(f'loss = {loss}, per_ch_score = {per_ch_score}')

            # print(f'bf output type = {output.type()}, output size = {output.size()}, ')
            output = output.squeeze()
            # loss, per_ch_score = criterion(output, target)

            # print(f'af output size = {output.size()}')
            output_arr = output.cpu().numpy()
            print(f'output_arr type = {type(output_arr)}, output_arr size = {np.shape(output_arr)}')
            print(f'output_arr min = {np.min(output_arr)}, output_arr max = {np.max(output_arr)}')
            file_name = f'pred_20_40_07_08_{batch_idx}.nii.gz'

            # os.chdir('E:/HSE/Medical_Segmentation/saved_models/UNET3D_checkpoints/UNET3D_17_08___07_06_thyroid_/prediction/')
            # # os.mkdir('prediction/')
            # print(f'current: {os.getcwd()}')
            # os.mkdir('prediction/')
            # output_arr = np.where(output_arr > 0, 1, (np.where(output_arr < -3, 0, 1)))

            # set threshold to the predicted image
            # output_arr = np.where(output_arr > 0, 1, 0)

            output_img = sitk.GetImageFromArray(output_arr[:, :, :])
            # print(f'output_img type = {type(output_img)}, output_img size = {output_img.size()}')
            os.chdir(path_list[batch_idx])
            sitk.WriteImage(output_img[:, :, :], file_name)
            print(f'{file_name} saved in {os.getcwd()}')
            print(f'prediction done -------------------------------\n')
            # print(f'output type = {output.type()}, output size = {output.size()}')
        break

            # loss, per_ch_score = self.criterion(output, target)


_, _, pred_loader = dataloaders.thyroid_dataloader.generate_thyroid_dataset()
PATH = 'E:/HSE/Medical_Segmentation/saved_models/UNET3D_checkpoints/'
# model_path = PATH + 'UNET3D_29_06___17_24_thyroid_/UNET3D_29_06___17_24_thyroid__BEST.pth'
# model_path = PATH + 'UNET3D_29_06___17_24_thyroid_/UNET3D_29_06___17_24_thyroid__last_epoch.pth'

predictor(PATH=PATH, data_loader=pred_loader)


# def predict_dataset(self, dataset, export_path):
#     """
#     Predicts the images in the given dataset and saves it to disk.
#     Args:
#         dataset: the dataset of images to be exported, instance of unet.dataset.Image2D
#         export_path: path to folder where results to be saved
#     """
#     self.net.train(False)
#     chk_mkdir(export_path)
#
#     for batch_idx, (X_batch, *rest) in enumerate(DataLoader(dataset, batch_size=1)):
#         if isinstance(rest[0][0], str):
#             image_filename = rest[0][0]
#         else:
#             image_filename = '%s.png' % str(batch_idx + 1).zfill(3)
#
#         X_batch = Variable(X_batch.to(device=self.device))
#         y_out = self.net(X_batch).cpu().data.numpy()
#
#         io.imsave(os.path.join(export_path, image_filename), y_out[0, 1, :, :])