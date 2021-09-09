import torch.nn as nn
import torch

"""
Loss function for Co learning network

"""


def tversky(pred, target):
    true_pos = (pred * target).sum(dim=1).sum(dim=1).sum(dim=1)
    false_neg = (target * (1-pred)).sum(dim=1).sum(dim=1).sum(dim=1)
    false_pos = ((1-target)*pred).sum(dim=1).sum(dim=1).sum(dim=1)

    alpha = 0.7
    return (true_pos + 1e-5)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + 1e-5)


class Co_DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, pred_temp, target):
        target = target.cuda()

        # similarity between two prediction results

        pred = pred.squeeze(dim=1)
        pred_temp = pred_temp.squeeze(dim=1)

        Similarity = (pred - pred_temp)
        SimilarityLoss = (torch.abs(Similarity)).mean()

        dice = tversky(pred, target)

        # dice_temp definition
        # dice_temp = 2 * (pred_temp * target).sum(dim=1).sum(dim=1).sum(dim=1) / (pred_temp.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
        #                                     target.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + 1e-5)

        # print(dice, dice_temp, SimilarityLoss)
        dice_temp = tversky(pred_temp, target)

        final_loss = ((1-dice)*0.5 + (1-dice_temp)*0.4).mean() + SimilarityLoss * 0.5

        return final_loss
