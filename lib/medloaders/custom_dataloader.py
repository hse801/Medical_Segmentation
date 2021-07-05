import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

import lib.augment3D as augment3D
import lib.utils as utils
from lib.medloaders import medical_image_process as img_loader
from lib.medloaders.medical_loader_utils import create_sub_volumes


class Thyroid(Dataset):
    """
    Code for reading the infant brain MICCAIBraTS2018 challenge
    """
    # split idx: for spliting training set / validation set

    def __init__(self, args, mode, dataset_path='E:/HSE/', classes=1, crop_dim=(128, 128, 128), split_idx=260,
                 samples=10,
                 load=False):
        """
        :param mode: 'train','val','test'
        :param dataset_path: root dataset folder
        :param crop_dim: subvolume tuple
        :param split_idx: 1 to 10 values
        :param samples: number of sub-volumes that you want to create
        """
        self.mode = mode
        self.root = str(dataset_path)
        self.training_path = self.root + 'Thyroid/Dicom/Train/'
        self.testing_path = self.root + 'Thyroid/Dicom/Test/'
        self.full_vol_dim = (128, 128, 128)  # slice, width, height
        self.crop_size = crop_dim
        self.threshold = args.threshold
        self.normalization = args.normalization
        self.augmentation = args.augmentation
        self.list = []
        self.samples = samples
        self.full_volume = None
        self.classes = classes

        if self.augmentation:
            self.transform = augment3D.RandomChoice(
                transforms=[augment3D.GaussianNoise(mean=0, std=0.01), augment3D.RandomFlip(),
                            augment3D.ElasticTransform()], p=0.5)
        self.save_name = self.root + '/Medical_Segmentation/Thyroid-list-' + mode + '-samples-' + str(samples) + '.txt'

        if load:
            ## load pre-generated data
            self.list = utils.load_list(self.save_name)
            ct_list = sorted(glob.glob(os.path.join(self.training_path, '*/CT_rsmpl.nii.gz')))
            self.affine = img_loader.load_affine_matrix(ct_list[0])
            return

        subvol = '_vol_' + str(crop_dim[0]) + 'x' + str(crop_dim[1]) + 'x' + str(crop_dim[2])
        self.sub_vol_path = self.root + '/brats2019/MICCAI_BraTS_2019_Data_Training/generated/' + mode + subvol + '/'
        utils.make_dirs(self.sub_vol_path)

        ct_list = sorted(glob.glob(os.path.join(self.training_path, '*/CT_rsmpl.nii.gz')))
        labels = sorted(glob.glob(os.path.join(self.training_path, '*/Mask_rsmpl.nii.gz')))
        ct_list, labels = utils.shuffle_lists(ct_list, labels, seed=17)
        self.affine = img_loader.load_affine_matrix(ct_list[0])

        # Total data: patient 308
        # For training: 258
        # For validation: 50
        if self.mode == 'train':
            print('Brats2019, Total data:', len(ct_list))
            ct_list = ct_list[:split_idx]
            labels = labels[:split_idx]
            self.list = create_sub_volumes(ct_list, labels,
                                           dataset_name="thyroid", mode=mode, samples=samples,
                                           full_vol_dim=self.full_vol_dim, crop_size=self.crop_size,
                                           sub_vol_path=self.sub_vol_path, th_percent=self.threshold)

        elif self.mode == 'val':
            ct_list = ct_list[split_idx:]
            labels = labels[split_idx:]
            self.list = create_sub_volumes(ct_list, labels,
                                           dataset_name="thyroid", mode=mode, samples=samples,
                                           full_vol_dim=self.full_vol_dim, crop_size=self.crop_size,
                                           sub_vol_path=self.sub_vol_path, th_percent=self.threshold)
        elif self.mode == 'test':
            self.list_IDsT1 = sorted(glob.glob(os.path.join(self.testing_path, '*/CT_rsmpl.nii.gz')))
            self.labels = None
            # Todo inference code here

        utils.save_list(self.save_name, self.list)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        f_t1, f_t1ce, f_t2, f_flair, f_seg = self.list[index]
        img_t1, img_t1ce, img_t2, img_flair, img_seg = np.load(f_t1), np.load(f_t1ce), np.load(f_t2), np.load(
            f_flair), np.load(f_seg)
        if self.mode == 'train' and self.augmentation:
            [img_t1, img_t1ce, img_t2, img_flair], img_seg = self.transform([img_t1, img_t1ce, img_t2, img_flair],
                                                                            img_seg)

            return torch.FloatTensor(img_t1.copy()).unsqueeze(0), torch.FloatTensor(img_t1ce.copy()).unsqueeze(
                0), torch.FloatTensor(img_t2.copy()).unsqueeze(0), torch.FloatTensor(img_flair.copy()).unsqueeze(
                0), torch.FloatTensor(img_seg.copy())

        return img_t1, img_t1ce, img_t2, img_flair, img_seg
