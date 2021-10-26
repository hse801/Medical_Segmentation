import numpy as np
import torch
from torch import nn as nn
from lib.utils.general import prepare_input
from lib.visual3D_temp.BaseWriter import TensorboardWriter
from lib.losses3D import *


class BCELossEdge(nn.Module):
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(BCELossEdge, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss()

    def weighted_BCE_cross_entropy(self, output, target, weights=None):
        if weights is not None:
            assert len(weights) == 2
            output = torch.clamp(output, min=1e-3, max=1-1e-3)
            bce = weights[1] * (target * torch.log(output)) + \
                  weights[0] * ((1-target) * torch.log((1-output)))
        else:
            output = torch.clamp(output, min=1e-3, max=1 - 1e-3)
            bce = target * torch.log(output) + (1-target) * torch.log((1-output))
        return torch.neg(torch.mean(bce))

    def forward(self, predict, target):

        bs, category, depth, width, height = target.shape
        bce_loss = []
        for i in range(predict.shape[1]):
            pred_i = predict[:, i]
            targ_i = target[:, i]
            tt = np.log(depth * width * height / (target[:, i].cpu().data.numpy().sum()+1))
            bce_i = self.weighted_BCE_cross_entropy(pred_i, targ_i, weights=[1, tt])
            bce_loss.append(bce_i)

        bce_loss = torch.stack(bce_loss)
        total_loss = bce_loss.mean()
        # print(f'loss.py: bce_loss = {bce_loss}, total_loss = {total_loss}')
        return total_loss