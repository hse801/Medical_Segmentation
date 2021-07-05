import glob
import os


def dcm2niis():
    file_list = glob.glob('E:/HSE/Thyroid/Dicom/*/*.dcm')
    os.chdir('C:/Users/Bohye/Desktop/mricron')
    for i in file_list[115:120]:
        os.system('dcm2nii.exe ' + i)

dcm2niis()