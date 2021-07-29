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


# Dataset
class Thyroid_dataset(Dataset):
    def __init__(self, ct_path, mask_path=None, test_flag=0, transform=None, left_path=None, right_path=None,
                 ConResNet=False, lr_flip=False):

        self.ct_path = ct_path
        self.mask_path = mask_path
        self.test_flag = test_flag
        self.transform = transform
        self.left_path = left_path
        self.right_path = right_path
        self.ConResNet = ConResNet
        self.lr_flip = lr_flip

    def __getitem__(self, idx):

        img_ct_path = self.ct_path[idx]
        img_ct = sitk.ReadImage(img_ct_path)
        # print(f'ct path for training = {img_ct_path}')
        img_ct_data = sitk.GetArrayFromImage(img_ct)
        img_ct_data = (img_ct_data - np.mean(img_ct_data)) / (np.std(img_ct_data) + 1e-8)
        # img_ct_data = img_ct_data.reshape(1, -1, 128, 128)
        # img_ct_data[img_ct_data > 500] = 500
        # torch.FloatTensor(img_ct_data.copy()).unsqueeze(0)
        # print(f'before squeeze ct shape = {img_ct_data.shape}')
        # img_ct_data = img_ct_data.unsqueeze(0)

        # for 1 channel
        img_mask_path = self.mask_path[idx]
        img_mask = sitk.ReadImage(img_mask_path)
        img_mask_data = sitk.GetArrayFromImage(img_mask)
        img_mask_data[img_mask_data > 0] = 1

        # if self.lr_flip:
        #     img_ct_data = img_ct_data[:, ::-1, :]
        #     img_mask_data = img_mask_data[:, ::-1, :]

        # For ConResNet
        # img -> res
        ct_size = np.shape(img_ct_data)[0]
        # print(f'bf img_ct_data shape = {np.shape(img_ct_data)}')
        # img_ct_data = np.reshape(img_ct_data, (1, ct_size, ct_size, ct_size))
        # print(f'af img_ct_data shape = {np.shape(img_ct_data)}')
        ct_copy = np.zeros((ct_size, ct_size, ct_size)).astype(np.float32)
        # print(f'ct_copy shape = {np.shape(ct_copy)}')
        ct_copy[1:, :, :] = img_ct_data[0: ct_size - 1, :, :]
        ct_res = img_ct_data - ct_copy
        ct_res[0, :, :] = 0

        # label -> res
        # img_mask_data = np.reshape(img_mask_data, (1, ct_size, ct_size, ct_size))
        mask_copy = np.zeros((ct_size, ct_size, ct_size)).astype(np.float32)
        mask_copy[1:, :, :] = img_mask_data[0: ct_size - 1, :, :]
        mask_res = img_mask_data - mask_copy

        # print(f'mask path for training = {img_mask_path}')

        # img_mask_data[img_mask_data == 5120] = 1 # for 2 labels
        # img_mask_data[img_mask_data > 5120] = 2

        # for 2 channel output
        # img_left_path = self.left_path[idx]
        # img_left = sitk.ReadImage(img_left_path)
        # img_left_data = sitk.GetArrayFromImage(img_left)
        #
        # img_right_path = self.right_path[idx]
        # img_right = sitk.ReadImage(img_right_path)
        # img_right_data = sitk.GetArrayFromImage(img_right)
        #
        # mask_combined = np.stack((img_left_data, img_right_data), axis=0)


        # mask_left = np.where(img_mask_data == 5120, 1, 0)
        # mask_right = np.where(img_mask_data == 7168, 1, 0)
        # # print(f'mask_left shape = {np.shape(mask_left)}, mask_right shape = {np.shape(mask_right)}')
        # mask_combined = np.stack((mask_left, mask_right), axis=0)
        # print(f'mask_combined.size = {np.shape(mask_combined)}')

        # print(f'mask max = {np.max(img_mask_data)}')
        # img_mask_data = img_mask_data / img_mask_data.max()
        # print(f'torch.FloatTensor(img_ct_data.copy()).unsqueeze(0) size = {torch.FloatTensor(img_ct_data.copy()).unsqueeze(0).size()}')
        # print(f'torch.FloatTensor(img_mask_data.copy()).unsqueeze(0) size = {torch.FloatTensor(img_mask_data.copy()).unsqueeze(0).size()}')
        # torch.FloatTensor(img_mask_data.copy()).unsqueeze(0) size = torch.Size([1, 128, 128, 128])

        if self.test_flag == 1:
            if self.lr_flip:
                img_ct_data = img_ct_data[:, ::-1, :]
                img_mask_data = img_mask_data[:, ::-1, :]

            return torch.FloatTensor(img_ct_data.copy()).unsqueeze(0), torch.FloatTensor(img_mask_data.copy()).unsqueeze(0)

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
        # transforms = tio.Compose([
        #     tio.
        # ])
        # print(f'ct type = {type(img_ct_data)}, mask type = {type(img_mask_data)}')
        # print(f'bf ct shape = {img_ct_data.shape}, mask shape = {img_mask_data.shape}')

        [img_ct_data], img_mask_data = self.transform([img_ct_data], img_mask_data)

        # [img_ct_data], mask_combined = self.transform([img_ct_data], mask_combined)

        # print(f'af ct shape = {img_ct_data.shape}, mask shape = {img_mask_data.shape}')
        # img_ct_data, mask_combined = self.transform(img_ct_data, mask_combined)

        # img_ct_dataimg_ct_data
        # print(f'torch.FloatTensor(img_mask_data.copy()) = {torch.FloatTensor(img_mask_data.copy()).size()}')
        # print(f'torch.FloatTensor(img_mask_data.copy()).unsqueeze(0) = {torch.FloatTensor(img_mask_data.copy()).unsqueeze(0).size()}')
        # print(f'torch.FloatTensor(img_ct_data.copy()).unsqueeze(0) = {torch.FloatTensor(img_ct_data.copy()).size()}')
        # print(f'torch.FloatTensor(img_ct_data.copy()) = {torch.FloatTensor(img_ct_data.copy()).unsqueeze(0).size()}')

        # return torch.FloatTensor(img_ct_data.copy()).unsqueeze(0), torch.FloatTensor(mask_combined.copy())
        # if self.ConResNet:
        #     return img_ct_data.copy(), ct_res.copy(), img_mask_data.copy(), mask_res.copy()

        return torch.FloatTensor(img_ct_data.copy()).unsqueeze(0), torch.FloatTensor(img_mask_data.copy()).unsqueeze(0)
        # return torch.FloatTensor(img_ct_data.copy()), torch.FloatTensor(img_mask_data.copy())

    def __len__(self):
        return len(self.ct_path)


"""Loading DATA"""

# 128x128x128
ct_path = glob.glob('E:/HSE/Thyroid/Dicom/*/CT_rsmpl.nii.gz')
mask_path = glob.glob('E:/HSE/Thyroid/Dicom/*/Mask_rsmpl.nii.gz')
# 64x64x64
crop_ct_path = glob.glob('E:/HSE/Thyroid/Dicom/*/crop_ct.nii.gz')
crop_mask_path = glob.glob('E:/HSE/Thyroid/Dicom/*/crop_mask.nii.gz')
# 64x64x64
crop_ct_size_path = glob.glob('E:/HSE/Thyroid/Dicom/*/crop_ct_size*.nii.gz')
crop_mask_size_path = glob.glob('E:/HSE/Thyroid/Dicom/*/crop_mask_size*.nii.gz')
# 64x64x64, left and right each
left_mask_path = glob.glob('E:/HSE/Thyroid/Dicom/*/crop_mask_left.nii.gz')
right_mask_path = glob.glob('E:/HSE/Thyroid/Dicom/*/crop_mask_right.nii.gz')


# train_ds = Thyroid_dataset(ct_path[0:308], mask_path[0:308], test_flag=0)
# val_ds = Thyroid_dataset(ct_path[308:368], mask_path[308:368], test_flag=1)

# train_ds = Thyroid_dataset(crop_ct_path[60:368], test_flag=0, left_path=left_mask_path[60:368], right_path=right_mask_path[60:368])
# val_ds = Thyroid_dataset(crop_ct_path[0:60], test_flag=1, left_path=left_mask_path[0:60], right_path=right_mask_path[0:60])
# pred_ds = Thyroid_dataset(crop_ct_path[0:101], test_flag=1, left_path=left_mask_path[0:101], right_path=right_mask_path[0:101])
# pred_ds = Thyroid_dataset(crop_ct_path[0:368], mask_path=crop_mask_path[0:368], test_flag=1)
# train_ds = Thyroid_dataset(crop_ct_path[60:368], mask_path=crop_mask_path[60:368], test_flag=0)
# val_ds = Thyroid_dataset(crop_ct_path[0:60], mask_path=crop_mask_path[0:60], test_flag=1)
# pred_ds = Thyroid_dataset(crop_ct_path[0:368], right_mask_path[0:368], test_flag=1)

# left mask
# train_ds = Thyroid_dataset(crop_ct_path[60:368], mask_path=left_mask_path[60:368], test_flag=0)
# val_ds = Thyroid_dataset(crop_ct_path[0:60], mask_path=left_mask_path[0:60], test_flag=1)
# right mask
train_ds = Thyroid_dataset(crop_ct_path[60:368], mask_path=right_mask_path[60:368], test_flag=0)
val_ds = Thyroid_dataset(crop_ct_path[0:60], mask_path=right_mask_path[0:60], test_flag=1)

# pred_ds = Thyroid_dataset(crop_ct_path[0:101], mask_path=right_mask_path[0:101], test_flag=1)
pred_ds = Thyroid_dataset(crop_ct_path[0:60], mask_path=left_mask_path[0:60], test_flag=1, lr_flip=True)

res_train_ds = Thyroid_dataset(crop_ct_path[60:368], mask_path=crop_mask_path[60:368], test_flag=0, ConResNet=True)
res_val_ds = Thyroid_dataset(crop_ct_path[0:60], mask_path=crop_mask_path[0:60], test_flag=1, ConResNet=True)


def generate_thyroid_dataset():

    train_loader = DataLoader(train_ds, batch_size=2, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=4, num_workers=4)
    # train_loader = DataLoader(res_train_ds, batch_size=2, num_workers=4)
    # val_loader = DataLoader(res_val_ds, batch_size=2, num_workers=4)
    pred_loader = DataLoader(pred_ds, batch_size=1, num_workers=0)

    return train_loader, val_loader, pred_loader
