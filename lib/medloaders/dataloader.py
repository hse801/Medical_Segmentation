import os
import torch
from torch.utils.data import Dataset
import glob
import numpy as np

import lib.utils as utils
#from lib.medloaders import img_loader
from lib.medloaders import medical_image_process as img_loader


class IXIMRIdataset(Dataset):
    """
    Code for reading the IXI brain MRI dataset
    This loader is implemented for cross-dataset testing
    """

    def __init__(self, args, dataset_path='E:/HSE/', voxels_space=(2, 2, 2), modalities=2, to_canonical=False,
                 save=True):
        """
        :param dataset_path: the extracted path that contains the desired images
        :param voxels_space: for reshampling the voxel space
        :param modalities: 1 for T1 only, 2 for T1 and T2
        :param to_canonical: If you want to convert the coordinates to RAS
        for more info on this advice here https://www.slicer.org/wiki/Coordinate_systems
        :param save: to save the generated data offline for faster reading
        and not load RAM
        """
        self.root = str(dataset_path)
        self.modalities = modalities
        self.training_path = self.root + 'Thyroid/Dicom/Train/'
        self.testing_path = self.root + 'Thyroid/Dicom/Test/'
        self.save = save
        self.CLASSES = 1
        self.full_vol_dim = (128, 128, 128)  # slice, width, height
        self.voxels_space = voxels_space
        self.modalities = str(modalities)
        self.list = []
        self.full_volume = None
        self.to_canonical = to_canonical
        self.affine = None

        subvol = '_vol_' + str(self.voxels_space[0]) + 'x' + str(self.voxels_space[1]) + 'x' + str(
            self.voxels_space[2])

        if self.save:
            self.sub_vol_path = self.root + '/ixi/generated/' + subvol + '/'
            utils.make_dirs(self.sub_vol_path)
        print(self.training_path)
        self.ct_list = sorted(glob.glob(os.path.join(self.training_path, '*/CT_rsmpl.nii.gz')))
        self.create_input_data()

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        # offline data
        if self.save:
            t1_path, t2_path = self.list[index]
            t1 = torch.from_numpy(np.load(t1_path))
            t2 = torch.from_numpy(np.load(t2_path))
            return t1, t2
        # on-memory saved data
        else:
            return self.list[index]

    def create_input_data(self):
        total = len(self.ct_list)
        print('Dataset samples: ', total)
        for i in range(total):
            print(i)
            ct_tensor = img_loader.load_medical_image(self.ct_list[i], type="ct", resample=self.voxels_space,
                                                          to_canonical=self.to_canonical)

        if self.save:
            filename = self.sub_vol_path + 'id_' + str(i) + '_s_' + str(i) + '_'
            f_ct = filename + 'CT_rsmpl.npy'
            np.save(f_ct, ct_tensor)
            self.list.append(f_ct)
        else:
            self.list.append(ct_tensor)
