import SimpleITK as sitk
import glob
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from torch.autograd import Variable
from torchvision import transforms, utils
import lib.augment3D as augment3D
import torchio as tio
from skimage.transform import resize
import os


# Dataset
class Lung_dataset(Dataset):
    def __init__(self, ct_path, pet_path, folder_path, test_flag=0):

        self.ct_path = ct_path
        self.pet_path = pet_path
        self.folder_path = folder_path
        self.test_flag = test_flag

    def __getitem__(self, idx):

        img_ct_path = self.ct_path[idx]
        img_ct = sitk.ReadImage(img_ct_path)
        # print(f'ct path for training = {img_ct_path}')
        img_ct_data = sitk.GetArrayFromImage(img_ct)
        img_ct_data = (img_ct_data - np.mean(img_ct_data)) / (np.std(img_ct_data) + 1e-8)
        # img_ct_data = img_ct_data.reshape(1, -1, 128, 128)
        # img_ct_data[img_ct_data > 500] = 500
        # torch.FloatTensor(img_ct_data.copy()).unsqueeze(0)
        # img_ct_data = img_ct_data.unsqueeze(0)

        img_pet_path = self.pet_path[idx]
        img_pet = sitk.ReadImage(img_pet_path)
        img_pet_data = sitk.GetArrayFromImage(img_pet)
        img_pet_data = (img_pet_data - np.mean(img_pet_data)) / (np.std(img_pet_data) + 1e-8)
        # img_pet_data = img_pet_data.reshape(1, -1, 128, 160)

        pet_ct_data = np.stack((img_ct_data, img_pet_data), axis=0)

        folder_path = self.folder_path[idx]
        primary_path = folder_path + 'ROI_cut.nii.gz'
        lymph_path = folder_path + 'lymph_cut_sum.nii.gz'

        if os.path.isfile(primary_path):
            img_primary = sitk.ReadImage(primary_path)
            img_primary_data = sitk.GetArrayFromImage(img_primary)
            # img_primary_data = img_primary_data.reshape(1, -1, 128, 160)
        else:
            img_primary_data = np.zeros((80, 128, 160))

        if os.path.isfile(lymph_path):
            img_lymph = sitk.ReadImage(lymph_path)
            img_lymph_data = sitk.GetArrayFromImage(img_lymph)
            # img_lymph_data = img_lymph_data.reshape(1, -1, 128, 160)
        else:
            img_lymph_data = np.zeros((80, 128, 160))
        # print(f'img_ct_data = {img_ct_data.shape}, img_pet_data = {img_pet_data.shape}')
        # print(f'img_primary_data = {img_primary_data.shape}, img_lymph_data = {img_lymph_data.shape}')

        # 2 class labels
        img_mask_data = np.stack((img_primary_data, img_lymph_data), axis=0)

        # 1 class labels
        # img_mask_data = img_primary_data + img_lymph_data
        # img_mask_data[img_mask_data > 0] = 1

        # sum lymph and primary
        # img_mask_data = img_primary_data + img_lymph_data
        # print(f'img_ct_data = {img_ct_data.shape}, img_pet_data = {img_pet_data.shape}, img_mask_data = {img_mask_data.shape}')
        if self.test_flag == 1:
            # return torch.FloatTensor(img_ct_data.copy()).unsqueeze(0), torch.FloatTensor(img_pet_data.copy()).unsqueeze(0),\
            #        torch.FloatTensor(img_mask_data.copy())
            return torch.FloatTensor(pet_ct_data.copy()), torch.FloatTensor(img_mask_data.copy())

        # if self.augmentation:
        # self.transform = augment3D.RandomChoice(
        #     transforms=[augment3D.RandomRotation(),
        #                 augment3D.RandomShift(), augment3D.RandomZoom()
        #                 ], p=0.4)
        # augment3D.GaussianNoise(mean=0, std=0.01),
        # [pet_ct_data], img_mask_data = self.transform([pet_ct_data], img_mask_data)

        # Apply transform
        # [img_ct_data, img_pet_data], img_mask_data = self.transform([img_ct_data, img_pet_data], img_mask_data)

        # pet_ct_data = np.stack((img_ct_data, img_pet_data), axis=0)

        # return torch.FloatTensor(img_ct_data.copy()).unsqueeze(0), torch.FloatTensor(img_pet_data.copy()).unsqueeze(0),\
        #        torch.FloatTensor(img_mask_data.copy())
        return torch.FloatTensor(pet_ct_data.copy()), torch.FloatTensor(img_mask_data.copy())

    def __len__(self):
        return len(self.ct_path)


"""Loading DATA"""

# 128x128x128
train_ct_path = glob.glob('E:/HSE/LungCancerData/train/*/CT_cut.nii.gz')
train_pet_path = glob.glob('E:/HSE/LungCancerData/train/*/PET_cut.nii.gz')
# primary_path = glob.glob('E:/HSE/LungCancerData/train/*/ROI_cut.nii.gz')
valid_ct_path = glob.glob('E:/HSE/LungCancerData/valid/*/CT_cut.nii.gz')
valid_pet_path = glob.glob('E:/HSE/LungCancerData/valid/*/PET_cut.nii.gz')
train_folder_path = glob.glob('E:/HSE/LungCancerData/train/*/')
valid_folder_path = glob.glob('E:/HSE/LungCancerData/valid/*/')


train_ds = Lung_dataset(train_ct_path[0:780], train_pet_path[0:780], train_folder_path[0:780], test_flag=0)
val_ds = Lung_dataset(valid_ct_path[0:70], valid_pet_path[0:70], valid_folder_path[0:70], test_flag=1)
pred_ds = Lung_dataset(valid_ct_path[0:70], valid_pet_path[0:70], valid_folder_path[0:70], test_flag=1)


def generate_lung_dataset():

    train_loader = DataLoader(train_ds, batch_size=2, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=4, num_workers=4)
    # train_loader = DataLoader(res_train_ds, batch_size=2, num_workers=4)
    # val_loader = DataLoader(res_val_ds, batch_size=2, num_workers=4)
    pred_loader = DataLoader(pred_ds, batch_size=1, num_workers=0)

    return train_loader, val_loader, pred_loader
