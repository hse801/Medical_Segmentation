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
        img_mask_data = np.stack((img_primary_data, img_lymph_data), axis=0)

        if self.test_flag == 1:
            return torch.FloatTensor(pet_ct_data.copy()), torch.FloatTensor(img_mask_data.copy())

        # else:
        #     img_ct_path = self.ct_path[idx]
        #     img_ct = sitk.ReadImage(img_ct_path)
        #     # print(f'ct path for validation = {img_ct_path}')
        #
        #     img_ct_data = sitk.GetArrayFromImage(img_ct)
        #     # img_ct_data = img_ct_data.reshape(1, -1, 128, 160)
        #     # img_ct_data[img_ct_data > 500] = 500
        #     # img_ct_data = (img_ct_data - nums[0]) / (nums[1] + 1e-8)
        #     img_ct_data = (img_ct_data - np.mean(img_ct_data)) / (np.std(img_ct_data) + 1e-8)
        #
        #     # for 1 channel
        #     img_mask_path = self.mask_path[idx]
        #     img_mask = sitk.ReadImage(img_mask_path)
        #     img_mask_data = sitk.GetArrayFromImage(img_mask)
        #     img_mask_data[img_mask_data > 0] = 1
        #
        #     # For ConResNet
        #     # img -> res
        #     # ct_size = np.shape(img_ct_data)[0]
        #     # # img_ct_data = np.reshape(img_ct_data, (1, ct_size, ct_size, ct_size))
        #     # ct_size = img_ct_data[0].size()
        #     # ct_copy = np.zeros((ct_size, ct_size, ct_size)).astype(np.float32)
        #     # ct_copy[1:, :, :] = img_ct_data[0: ct_size - 1, :, :]
        #     # ct_res = img_ct_data - ct_copy
        #     # ct_res[0, :, :] = 0
        #
        #     # if self.ConResNet:
        #     #     return img_ct_data.copy(), ct_res.copy(), img_mask_data.copy()
        #
        #
        #
        #
        #     # print(f'mask path for validation = {img_mask_path}')
        #     # print(f'before  reshape ct shape = {img_ct_data.shape}, mask shape = {img_mask_data.shape}')
        #     # img_mask_data = img_mask_data.reshape(1, -1, 128, 160)
        #     # print(f'after reshape ct shape = {img_ct_data.shape}, mask shape = {img_mask_data.shape}')
        #
        #     # for 2 channel output
        #     # img_left_path = self.left_path[idx]
        #     # img_left = sitk.ReadImage(img_left_path)
        #     # img_left_data = sitk.GetArrayFromImage(img_left)
        #     #
        #     # img_right_path = self.right_path[idx]
        #     # img_right = sitk.ReadImage(img_right_path)
        #     # img_right_data = sitk.GetArrayFromImage(img_right)
        #     #
        #     # mask_combined = np.stack((img_left_data, img_right_data), axis=0)
        #
        #     # create 2 channel mask
        #     # mask_left = np.where(img_mask_data == 5120, 1, 0)
        #     # mask_right = np.where(img_mask_data == 7168, 1, 0)
        #     # mask_combined = np.stack((mask_left, mask_right), axis=0)
        #     # print(f'mask_combined.size = {np.shape(mask_combined)}')
        #
        #     # img_mask_data[img_mask_data == 5120] = 1 # for 2 labels
        #     # img_mask_data[img_mask_data > 5120] = 2
        #
        #     # img_mask_data = img_mask_data / img_mask_data.max()
        #
        #     # return torch.FloatTensor(img_ct_data.copy()).unsqueeze(0), torch.FloatTensor(mask_combined.copy())
        #     return torch.FloatTensor(img_ct_data.copy()).unsqueeze(0), torch.FloatTensor(img_mask_data.copy()).unsqueeze(0)
        #     # return torch.FloatTensor(img_ct_data.copy()), torch.FloatTensor(img_mask_data.copy())
        # print(f'ct shape = {img_ct_data.shape}, mask shape = {img_mask_data.shape}')
        # print(f'ct max = {np.max(img_ct_data)}, mask max = {np.max(img_mask_data)}')
        # return img_ct_data, img_mask_data

        # if self.augmentation:
        self.transform = augment3D.RandomChoice(
            transforms=[augment3D.GaussianNoise(mean=0, std=0.01), augment3D.RandomRotation(),
                        augment3D.RandomShift(), augment3D.RandomZoom()
                        ], p=0.8)

        [pet_ct_data], img_mask_data = self.transform([pet_ct_data], img_mask_data)
        # [img_ct_data, img_pet_data], img_mask_data = self.transform([img_ct_data, img_pet_data], img_mask_data)

        return torch.FloatTensor(pet_ct_data.copy()), torch.FloatTensor(img_mask_data.copy())

    def __len__(self):
        return len(self.ct_path)


"""Loading DATA"""

# 128x128x128
ct_path = glob.glob('E:/HSE/LungCancerData/train/*/CT_cut.nii.gz')
pet_path = glob.glob('E:/HSE/LungCancerData/train/*/PET_cut.nii.gz')
# primary_path = glob.glob('E:/HSE/LungCancerData/train/*/ROI_cut.nii.gz')
folder_path = glob.glob('E:/HSE/LungCancerData/train/*/')


train_ds = Lung_dataset(ct_path[60:368], pet_path[60:368], folder_path[60:368], test_flag=0)
val_ds = Lung_dataset(ct_path[0:60], pet_path[0:60], folder_path[0:60], test_flag=1)
pred_ds = Lung_dataset(ct_path[0:60], pet_path[0:60], folder_path[0:60], test_flag=1)


def generate_lung_dataset():

    train_loader = DataLoader(train_ds, batch_size=2, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=4, num_workers=4)
    # train_loader = DataLoader(res_train_ds, batch_size=2, num_workers=4)
    # val_loader = DataLoader(res_val_ds, batch_size=2, num_workers=4)
    pred_loader = DataLoader(pred_ds, batch_size=1, num_workers=0)

    return train_loader, val_loader, pred_loader
