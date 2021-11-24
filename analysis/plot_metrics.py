import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd


ResUNet_path = 'E:/HSE/Medical_Segmentation/saved_models/RESUNETOG_checkpoints/RESUNETOG_18_41___10_16_thyroid_/'
UNet_path = 'E:/HSE/Medical_Segmentation/saved_models/UNET3D_checkpoints/UNET3D_02_43___07_14_thyroid_/'

ResNet_df = pd.read_csv(ResUNet_path + 'prediction_BEST_0902_2.csv')
UNet_df = pd.read_csv(UNet_path + 'prediction_BEST_0902.csv')

print(ResNet_df.head())
ResNet_df['dice_p'].plot(kind='hist')