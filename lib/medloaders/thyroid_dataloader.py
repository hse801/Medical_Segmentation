import SimpleITK as sitk
import glob
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from torch.autograd import Variable
from torchvision import transforms, utils
import lib.augment3D as augment3D
import torchio as tio


# Dataset
class Thyroid_dataset(Dataset):
    def __init__(self, ct_path, mask_path, test_flag=0, transform=None):

        self.ct_path = ct_path
        self.mask_path = mask_path
        self.test_flag = test_flag
        self.transform = transform

    def __getitem__(self, idx):

        if self.test_flag == 0:

            img_ct_path = self.ct_path[idx]
            img_ct = sitk.ReadImage(img_ct_path)
            # print(f'ct path for training = {img_ct_path}')
            # print(f'ct path = {img_ct_path}')
            img_ct_data = sitk.GetArrayFromImage(img_ct)
            img_ct_data = (img_ct_data - np.mean(img_ct_data)) / (np.std(img_ct_data) + 1e-8)
            # img_ct_data = img_ct_data.reshape(1, -1, 128, 128)
            # img_ct_data[img_ct_data > 500] = 500
            # torch.FloatTensor(img_ct_data.copy()).unsqueeze(0)
            # print(f'before squeeze ct shape = {img_ct_data.shape}')
            # img_ct_data = img_ct_data.unsqueeze(0)

            img_mask_path = self.mask_path[idx]
            img_mask = sitk.ReadImage(img_mask_path)
            img_mask_data = sitk.GetArrayFromImage(img_mask)
            # print(f'mask path for training = {img_mask_path}')

            # img_mask_data = img_mask_data.reshape(1, -1, 128, 128)
            # print(f'after ct shape = {img_ct_data.shape}')

            img_mask_data[img_mask_data > 0] = 1

            # img_mask_data[img_mask_data == 5120] = 1 # for 2 labels
            # img_mask_data[img_mask_data > 5120] = 2

            mask_left = np.where(img_mask_data == 5120, 1, 0)
            mask_right = np.where(img_mask_data == 7168, 1, 0)
            # print(f'mask_left shape = {np.shape(mask_left)}, mask_right shape = {np.shape(mask_right)}')
            mask_combined = np.stack((mask_left, mask_right), axis=0)
            # print(f'mask_combined.size = {np.shape(mask_combined)}')

            # print(f'mask max = {np.max(img_mask_data)}')
            # img_mask_data = img_mask_data / img_mask_data.max()
            # print(f'torch.FloatTensor(img_ct_data.copy()).unsqueeze(0) size = {torch.FloatTensor(img_ct_data.copy()).unsqueeze(0).size()}')
            # print(f'torch.FloatTensor(img_mask_data.copy()).unsqueeze(0) size = {torch.FloatTensor(img_mask_data.copy()).unsqueeze(0).size()}')
            # torch.FloatTensor(img_mask_data.copy()).unsqueeze(0) size = torch.Size([1, 128, 128, 128])

        else:
            # img_data_path = self.data_path[idx]
            # f = open(img_data_path, 'r')
            # nums = [float(x) for x in f.read().split()]
            # f.close()

            img_ct_path = self.ct_path[idx]
            img_ct = sitk.ReadImage(img_ct_path)
            # print(f'ct path for validation = {img_ct_path}')

            img_ct_data = sitk.GetArrayFromImage(img_ct)
            # img_ct_data = img_ct_data.reshape(1, -1, 128, 160)
            # img_ct_data[img_ct_data > 500] = 500
            # img_ct_data = (img_ct_data - nums[0]) / (nums[1] + 1e-8)
            img_ct_data = (img_ct_data - np.mean(img_ct_data)) / (np.std(img_ct_data) + 1e-8)

            img_mask_path = self.mask_path[idx]
            img_mask = sitk.ReadImage(img_mask_path)
            img_mask_data = sitk.GetArrayFromImage(img_mask)
            # print(f'mask path for validation = {img_mask_path}')
            # print(f'before  reshape ct shape = {img_ct_data.shape}, mask shape = {img_mask_data.shape}')
            # img_mask_data = img_mask_data.reshape(1, -1, 128, 160)
            # print(f'after reshape ct shape = {img_ct_data.shape}, mask shape = {img_mask_data.shape}')

            # for 1 label
            img_mask_data[img_mask_data > 0] = 1

            # create 2 channel mask
            mask_left = np.where(img_mask_data == 5120, 1, 0)
            mask_right = np.where(img_mask_data == 7168, 1, 0)
            mask_combined = np.stack((mask_left, mask_right), axis=0)
            # print(f'mask_combined.size = {np.shape(mask_combined)}')

            # img_mask_data[img_mask_data == 5120] = 1 # for 2 labels
            # img_mask_data[img_mask_data > 5120] = 2

            # img_mask_data = img_mask_data / img_mask_data.max()
            return torch.FloatTensor(img_ct_data.copy()).unsqueeze(0), torch.FloatTensor(mask_combined.copy())
            # return torch.FloatTensor(img_ct_data.copy()).unsqueeze(0), torch.FloatTensor(img_mask_data.copy()).unsqueeze(0)

        # print(f'ct shape = {img_ct_data.shape}, mask shape = {img_mask_data.shape}')
        # print(f'ct max = {np.max(img_ct_data)}, mask max = {np.max(img_mask_data)}')
        # return img_ct_data, img_mask_data

        # if self.augmentation:
        self.transform = augment3D.RandomChoice(
            transforms=[augment3D.GaussianNoise(mean=0, std=0.01),
                        ], p=0.3)
        # transforms = tio.Compose([
        #     tio.
        # ])
        # print(f'ct type = {type(img_ct_data)}, mask type = {type(img_mask_data)}')
        print(f'bf ct shape = {img_ct_data.shape}, mask shape = {img_mask_data.shape}')
        [img_ct_data], img_mask_data = self.transform([img_ct_data], img_mask_data)
        print(f'af ct shape = {img_ct_data.shape}, mask shape = {img_mask_data.shape}')
        # img_ct_data, mask_combined = self.transform(img_ct_data, mask_combined)

        # img_ct_dataimg_ct_data
        # print(f'torch.FloatTensor(img_mask_data.copy()) = {torch.FloatTensor(img_mask_data.copy()).size()}')
        # print(f'torch.FloatTensor(img_mask_data.copy()).unsqueeze(0) = {torch.FloatTensor(img_mask_data.copy()).unsqueeze(0).size()}')
        # print(f'torch.FloatTensor(img_ct_data.copy()).unsqueeze(0) = {torch.FloatTensor(img_ct_data.copy()).size()}')
        # print(f'torch.FloatTensor(img_ct_data.copy()) = {torch.FloatTensor(img_ct_data.copy()).unsqueeze(0).size()}')

        return torch.FloatTensor(img_ct_data.copy()).unsqueeze(0), torch.FloatTensor(mask_combined.copy())
        # return torch.FloatTensor(img_ct_data.copy()).unsqueeze(0), torch.FloatTensor(img_mask_data.copy()).unsqueeze(0)

        # if self.transform:
        #     if (self.test_flag == 0):
        #         img_ct_data, img_pet_data, img_mask_data = transform(img_ct_data, img_pet_data, img_mask_data, depth)
        #
        # img_ct_data = (img_ct_data - nums[0]) / (nums[1] + 1e-8)
        # img_pet_data = (img_pet_data - nums[2]) / (nums[3] + 1e-8)
        #
        # return img_ct_data, img_mask_data

    def __len__(self):
        return len(self.ct_path)


# transform = data_aug_new

"""Loading DATA"""

ct_path = glob.glob('E:/HSE/Thyroid/Dicom/*/CT_rsmpl.nii.gz')
mask_path = glob.glob('E:/HSE/Thyroid/Dicom/*/Mask_rsmpl.nii.gz')
crop_ct_path = glob.glob('E:/HSE/Thyroid/Dicom/*/crop_ct.nii.gz')
crop_mask_path = glob.glob('E:/HSE/Thyroid/Dicom/*/crop_mask.nii.gz')

# train_ds = Thyroid_dataset(ct_path[0:308], mask_path[0:308], test_flag=0)
# val_ds = Thyroid_dataset(ct_path[308:368], mask_path[308:368], test_flag=1)

train_ds = Thyroid_dataset(crop_ct_path[60:368], crop_mask_path[60:368], test_flag=0)
val_ds = Thyroid_dataset(crop_ct_path[0:60], crop_mask_path[0:60], test_flag=1)
pred_ds = Thyroid_dataset(ct_path[0:368], mask_path[0:368], test_flag=1)


def generate_thyroid_dataset():

    train_loader = DataLoader(train_ds, batch_size=2, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=4)
    pred_loader = DataLoader(pred_ds, batch_size=1, num_workers=0)

    return train_loader, val_loader, pred_loader
