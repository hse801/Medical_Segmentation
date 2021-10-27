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

"""

Prediction code for thyroid segmentation
with 1 input channel and 1 class
save metrics in.csv file

Also create nii file of the predicion image

"""


def predictor(PATH, data_loader, model_path, csv_name, save_folder):

    # path_list = glob.glob('E:/HSE/Thyroid/Dicom/*/')
    path_list = glob.glob('D:/0902_Thyroid/ThyroidSPECT Dataset/*/Tc Thyroid SPECT/')
    model_path = PATH + model_path
    # model = medzoo.UNet3D(in_channels=1, n_classes=1, base_n_filter=24)
    model = medzoo.ResidualUNet3D(in_channels=1, out_channels=1)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    val = 0.0
    eval_list = []
    dsc_sum = 0

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
            criterion = DiceLoss(classes=1)
            loss, dice_score = criterion(pred, label)
            print(f'loss = {loss:.4f}, dice_score = {dice_score[0]:.4f}')
            dsc_sum += dice_score[0]
            total_num = batch_idx + 1
            pred = pred.squeeze()
            label = label.squeeze()

            pred_arr = pred.cpu().numpy()
            label_arr = label.cpu().numpy()
            print(f'pred shape = {np.shape(pred_arr)}, label shape = {np.shape(label_arr)}')
            pred_arr = np.where(pred_arr > 0, 1, 0)

            thyroid_matrix = ConfusionMatrix(pred=pred_arr[:, :, :], label=label_arr[:, :, :])

            tp, fp, tn, fn = thyroid_matrix.get_matrix()
            print(f'Primary: tp = {tp}, fp = {fp}, tn = {tn}, fn = {fn}')

            recall = analysis.eval_metrics.recall(confusion_matrix=thyroid_matrix)
            print(f'recall = {recall:.4f}')

            precision = analysis.eval_metrics.precision(confusion_matrix=thyroid_matrix)
            print(f'precision p = {precision:.4f}')

            fscore = analysis.eval_metrics.f1_score(confusion_matrix=thyroid_matrix)
            print(f'f1 score p = {fscore:.4f}')

            hausdorff_distance = analysis.eval_metrics.hausdorff_distance(confusion_matrix=thyroid_matrix)
            print(f'hausdorff_distance p = {hausdorff_distance:.4f}')

            hausdorff_distance_95 = analysis.eval_metrics.hausdorff_distance_95(confusion_matrix=thyroid_matrix)
            print(f'hausdorff_distance 95 p = {hausdorff_distance_95:.4f}')
            eval_metrics = {}
            eval_metrics.update({'dice_p': dice_score,
                                 'recall': recall,
                                 'precision': precision,
                                 'fscore': fscore,
                                 'hausdorff_distance': hausdorff_distance,
                                 'hausdorff_distance_95': hausdorff_distance_95})
            eval_list.append(eval_metrics)


            # pred = pred.squeeze()
            # pred_arr = pred.cpu().numpy()
            # print(f'pred_arr type = {type(pred_arr)}, pred_arr size = {np.shape(pred_arr)}')
            print(f'pred_arr min = {np.min(pred_arr)}, pred_arr max = {np.max(pred_arr)}')

            file_name = f'pred_18_41_10_16_{batch_idx}.nii.gz'

            # print(f'output_img type = {type(output_img)}, output_img size = {output_img.size()}')
            os.chdir(path_list[batch_idx])
            pred_img = sitk.GetImageFromArray(pred_arr[:, :, :])
            sitk.WriteImage(pred_img[:, :, :], file_name)
            print(f'{file_name} saved in {os.getcwd()}')
            print(f'prediction done -------------------------------\n')
            # print(f'pred type = {pred.type()}, pred size = {pred.size()}')
        # break
    print(f'Evaluation dataframe: ')
    eval_df = pd.DataFrame(eval_list, columns=['dice_p', 'recall', 'precision', 'fscore',
                                            'hausdorff_distance', 'hausdorff_distance_95'])
    # Add row of Mean value of each metrics
    eval_df.loc['Mean'] = eval_df.mean()
    eval_df.loc['Median'] = eval_df.median()
    eval_df.loc['Std'] = eval_df.std()
    print(eval_df)
    print(eval_df.loc['Mean'])
    # os.chdir(PATH + save_folder)
    eval_df.to_csv(PATH + save_folder + csv_name, mode='w')
    print(f'Evaluation csv saved in {os.getcwd()}')
    print('End of validation')


_, _, pred_loader = dataloaders.thyroid_dataloader.generate_thyroid_dataset()
PATH = 'E:/HSE/Medical_Segmentation/saved_models/RESUNETOG_checkpoints/'
model_path = 'RESUNETOG_18_41___10_16_thyroid_/RESUNETOG_18_41___10_16_thyroid__BEST.pth'
save_folder = 'RESUNETOG_18_41___10_16_thyroid_/'
# save_folder = 'RESUNETOGT_1023_1235_thyroid/'
csv_name = 'prediction_BEST_0902_2.csv'
# model_path = PATH + 'UNET3D_29_06___17_24_thyroid_/UNET3D_29_06___17_24_thyroid__BEST.pth'
# model_path = PATH + 'UNET3D_29_06___17_24_thyroid_/UNET3D_29_06___17_24_thyroid__last_epoch.pth'

predictor(PATH=PATH, data_loader=pred_loader, model_path=model_path, csv_name=csv_name, save_folder=save_folder)