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
import analysis
from analysis import eval_metrics
from analysis.eval_metrics import ConfusionMatrix
import pandas as pd


def predictor(PATH, data_loader, model_path, csv_name, save_folder, mode='test'):

    # model_path = PATH + 'UNET3D_29_06___17_24_thyroid_/UNET3D_29_06___17_24_thyroid__BEST.pth'
    # 2 channel label, 2 classes
    model_path = PATH + model_path
    path_list = glob.glob(f'F:/LungCancerData/{mode}/*/')

    # model = medzoo.UNet3D(in_channels=2, n_classes=2, base_n_filter=24)
    model = medzoo.ResidualUNet3D(in_channels=2, out_channels=2)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # model = model.cuda()
    model.eval()
    val = 0.0
    primary_dice = []
    lymph_dice = []
    f1_p_list = []
    f1_l_list = []
    eval_list = []

    for batch_idx, input_tuple in enumerate(data_loader):
        with torch.no_grad():
            # input_tensor, label = prepare_input(input_tuple=input_tuple, args=self.args)
            input_tensor, label = input_tuple
            # print(f'input tuple len = {len(input_tuple)}, input tuple = {input_tuple}')
            # print(f'label type = {type(label)}, label = {label.nonzero()}')
            # input_tensor, label = input_tensor.cuda(), label.cuda()
            input_tensor, label = input_tensor.cpu(), label.cpu()
            input_tensor.requires_grad = False

            pred = model(input_tensor)
            criterion = DiceLoss(classes=2)
            print(f'pred size = {pred.size()}, label = {label.size()}')
            loss, dice = criterion(pred, label)
            print(f'loss = {loss}, dice = {dice}')
            primary_nonzero = label[:, 0, :, :, :].nonzero()
            lymph_nonzero = label[:, 1, :, :, :].nonzero()

            if primary_nonzero.nelement() == 0:
                print(f'No primary tumor')
                primary_dice.append(float('NaN'))
                lymph_dice.append(dice[0])
                print(f'DSC: {dice[0]:.4f}     Primary: None     Lymph: {dice[0]:.4f}')
            elif lymph_nonzero.nelement() == 0:
                print(f'No lymph node')
                primary_dice.append(dice[0])
                lymph_dice.append(float('NaN'))
                print(f'DSC: {dice[0]:.4f}     Primary: {dice[0]:.4f}     Lymph: None')
            else:
                # both primary and lymph exist
                primary_dice.append(dice[0])
                lymph_dice.append(dice[1])
                print(f'DSC: {dice.mean():.4f}     Primary: {dice[0]:.4f}     Lymph: {dice[1]:.4f}')

            # val += dice
            # print(f'DSC: {dice.mean():.4f}     Primary: {dice[0]:.4f}     Lymph: {dice[1]:.4f}')
            # print(f'pred size = {pred.size()}, label = {label.size()}')
            # pred size = torch.Size([1, 2, 80, 128, 160]), label = torch.Size([1, 2, 80, 128, 160])

            # hausdorff_dist = utils.metrics.compute_channel_hausdorff(pred, label)
            # hausdorff_dist = utils.metrics.hausdorff_distance(pred, label)
            # print(f'hausdorff distance = {hausdorff_dist}')
            pred_arr = pred.cpu().numpy()
            label_arr = label.cpu().numpy()
            pred_arr = np.where(pred_arr > 0.5, 1, 0)

            primary_matrix = ConfusionMatrix(pred=pred_arr[:, 0, :, :, :], label=label_arr[:, 0, :, :, :])
            lymph_matrix = ConfusionMatrix(pred=pred_arr[:, 1, :, :, :], label=label_arr[:, 1, :, :, :])

            tp, fp, tn, fn = primary_matrix.get_matrix()
            print(f'Primary: tp = {tp}, fp = {fp}, tn = {tn}, fn = {fn}')
            tp, fp, tn, fn = lymph_matrix.get_matrix()
            print(f'Lymph: tp = {tp}, fp = {fp}, tn = {tn}, fn = {fn}')

            recall_p = analysis.eval_metrics.recall(confusion_matrix=primary_matrix)
            recall_l = analysis.eval_metrics.recall(confusion_matrix=lymph_matrix)
            print(f'recall p = {recall_p:.4f}       recall l = {recall_l:.4f}')

            precision_p = analysis.eval_metrics.precision(confusion_matrix=primary_matrix)
            precision_l = analysis.eval_metrics.precision(confusion_matrix=lymph_matrix)
            print(f'precision p = {precision_p:.4f}     precision l = {precision_l:.4f}')

            fscore_p = analysis.eval_metrics.f1_score(confusion_matrix=primary_matrix)
            fscore_l = analysis.eval_metrics.f1_score(confusion_matrix=lymph_matrix)
            print(f'f1 score p = {fscore_p:.4f}     f1 score l = {fscore_l:.4f}')

            hausdorff_distance_p = analysis.eval_metrics.hausdorff_distance(confusion_matrix=primary_matrix)
            hausdorff_distance_l = analysis.eval_metrics.hausdorff_distance(confusion_matrix=lymph_matrix)
            print(f'hausdorff_distance p = {hausdorff_distance_p:.4f}       hausdorff_distance l = {hausdorff_distance_l:.4f}')

            hausdorff_distance_95_p = analysis.eval_metrics.hausdorff_distance_95(confusion_matrix=primary_matrix)
            hausdorff_distance_95_l = analysis.eval_metrics.hausdorff_distance_95(confusion_matrix=lymph_matrix)
            print(f'hausdorff_distance 95 p = {hausdorff_distance_95_p:.4f}       hausdorff_distance 95 l = {hausdorff_distance_95_l:.4f}')
            eval_metrics = {}
            eval_metrics.update({'dice_p': primary_dice[batch_idx], 'dice_l': lymph_dice[batch_idx],
                                 'recall_p': recall_p, 'recall_l': recall_l,
                                 'precision_p': precision_p, 'precision_l': precision_l,
                                 'fscore_p': fscore_p, 'fscore_l': fscore_l,
                                 'hausdorff_distance_p': hausdorff_distance_p, 'hausdorff_distance_l': hausdorff_distance_l,
                                 'hausdorff_distance_95_p': hausdorff_distance_95_p,
                                 'hausdorff_distance_95_l': hausdorff_distance_95_l})
            eval_list.append(eval_metrics)


            pred = pred.squeeze()
            output_arr = pred.cpu().numpy()
            print(f'output_arr type = {type(output_arr)}, output_arr size = {np.shape(output_arr)}')
            print(f'output_arr min = {np.min(output_arr)}, output_arr max = {np.max(output_arr)}')

            # file_name1 = f'pred_2ch_1_{batch_idx}.nii.gz'
            # file_name2 = f'pred_2ch_2_{batch_idx}.nii.gz'
            file_name = f'pred_09_27_09_16_{batch_idx}.nii.gz'

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
            # sitk.WriteImage(output_combined[:, :, :], file_name)
            print(f'{file_name} saved in {os.getcwd()}')
            print(f'prediction done -------------------------------\n')
            # print(f'pred type = {pred.type()}, pred size = {pred.size()}')
        # break
    print(f'Evaluation dataframe: ')
    eval_df = pd.DataFrame(eval_list, columns=['dice_p', 'dice_l', 'recall_p', 'recall_l', 'precision_p', 'precision_l',
                                               'fscore_p', 'fscore_l',
                                                'hausdorff_distance_p', 'hausdorff_distance_l',
                                                'hausdorff_distance_95_p', 'hausdorff_distance_95_l'])
    # Add row of Mean value of each metrics
    os.chdir(PATH + save_folder)
    eval_df.loc['Mean'] = eval_df.mean()
    print(eval_df)
    print(eval_df.loc['Mean'])
    eval_df.to_csv(PATH + csv_name, mode='w')
    print(f'Evaluation csv saved in {os.getcwd()}')


_, _, pred_loader = dataloaders.lung_dataloader.generate_lung_dataset()
PATH = 'E:/HSE/Medical_Segmentation/saved_models/RESUNETOGL_checkpoints/'
model_path = 'RESUNETOGL_09_27___09_16_lung_/RESUNETOGL_09_27___09_16_lung__BEST.pth'
save_folder = 'RESUNETOGL_09_27___09_16_lung_/'
csv_name = '/prediction_BEST.csv'
# model_path = PATH + 'UNET3D_29_06___17_24_thyroid_/UNET3D_29_06___17_24_thyroid__BEST.pth'
# model_path = PATH + 'UNET3D_29_06___17_24_thyroid_/UNET3D_29_06___17_24_thyroid__last_epoch.pth'

predictor(PATH=PATH, data_loader=pred_loader, model_path=model_path, csv_name=csv_name, save_folder=save_folder, mode='test')
