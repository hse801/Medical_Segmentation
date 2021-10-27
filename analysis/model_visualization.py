import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import cv2 as cv
import argparse
from torchvision import models, transforms
import lib.medzoo as medzoo

# ap = argparse.ArgumentParser()
# ap.add_argument('-i', '--image', required=True, help='path to image')
# args = vars(ap.parse_args())

# load the model
# model = models.resnet50(pretrained=True)

PATH = 'E:/HSE/Medical_Segmentation/saved_models/RESUNETOG_checkpoints/'
model_path = 'RESUNETOG_18_41___10_16_thyroid_/RESUNETOG_18_41___10_16_thyroid__BEST.pth'

model = medzoo.ResidualUNet3D(in_channels=1, out_channels=1)
# checkpoint = torch.load(PATH + model_path)
# model.load_state_dict(checkpoint['model_state_dict'])

# print(model)
model_weights = [] # we will save the conv layer weights in this list
conv_layers = [] # we will save the 49 conv layers in this list
# get all the model children as list
model_children = list(model.children())
print(f'model_children = {model_children}, len = {len(model_children)}')
for c in model_children:
    print(f'type = {type(c)}')


# counter to keep count of the conv layers
counter = 0
# append all the conv layers and their respective weights to the list
for i in range(len(model_children)):
    if type(model_children[i]) == torch.nn.modules.conv.Conv3d:
        counter += 1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == torch.nn.modules.container.ModuleList:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                print(f'child = {child}, len = {len(child)}')
                if type(child) == nn.Conv2d:
                    counter += 1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
print(f"Total convolutional layers: {counter}")














