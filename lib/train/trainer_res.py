import numpy as np
import torch
from torch import nn as nn
from lib.utils.general import prepare_input
from lib.visual3D_temp.BaseWriter import TensorboardWriter
from lib.losses3D import *
# from lib.losses3D import BCELossEdge


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


class Trainer_res:
    """
    Trainer class
    """

    def __init__(self, args, model, criterion, optimizer, train_data_loader,
                 valid_data_loader=None):

        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        # self.dice_score = dice_score
        self.train_data_loader = train_data_loader
        # epoch-based training
        self.len_epoch = len(self.train_data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        # self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
        self.log_step = int(np.sqrt(train_data_loader.batch_size))
        self.writer = TensorboardWriter(args)

        self.save_frequency = 10
        self.terminal_show_freq = self.args.terminal_show_freq
        self.start_epoch = 1
        self.bce_edge_loss = BCELossEdge()

    def training(self):
        for epoch in range(self.start_epoch, self.args.nEpochs):

            self.train_epoch(epoch)

            if self.do_validation:
                self.validate_epoch(epoch)

            val_loss = self.writer.data['val']['loss'] / self.writer.data['val']['count']

            if self.args.save is not None and ((epoch + 1) % self.save_frequency):
                self.model.save_checkpoint(self.args.save,
                                           epoch, val_loss,
                                           optimizer=self.optimizer)

            self.writer.write_end_of_epoch(epoch)
            self.writer.reset('train')
            self.writer.reset('val')

    def train_epoch(self, epoch):
        self.model.train()
        print('Training start---------------------------------------------------')
        for batch_idx, input_tuple in enumerate(self.train_data_loader):

            self.optimizer.zero_grad()
            input_tensor, target, target_edge = input_tuple
            input_tensor.requires_grad = True
            output = self.model(input_tensor)
            dice_loss, per_ch_score = self.criterion(output, target)
            edge_loss = self.bce_edge_loss.forward(target, target_edge)

            total_loss = dice_loss + edge_loss
            print(f'dice loss = {dice_loss}, edge_loss = {edge_loss}, total loss = {total_loss}')
            total_loss.backward()
            self.optimizer.step()
            # self.lr_scheduler.step()

            self.writer.update_scores(batch_idx, dice_loss.item(), per_ch_score, 'train',
                                      epoch * self.len_epoch + batch_idx)

            if (batch_idx + 1) % self.terminal_show_freq == 0:
                partial_epoch = epoch + batch_idx / self.len_epoch - 1
                self.writer.display_terminal(partial_epoch, epoch, 'train')

        self.writer.display_terminal(self.len_epoch, epoch, mode='train', summary=True)

    def validate_epoch(self, epoch):
        self.model.eval()

        for batch_idx, input_tuple in enumerate(self.valid_data_loader):
            with torch.no_grad():
                input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
                input_tensor.requires_grad = False
                output = self.model(input_tensor)

                loss, per_ch_score = self.criterion(output, target)

                self.writer.update_scores(batch_idx, loss.item(), per_ch_score, 'val',
                                          epoch * self.len_epoch + batch_idx)

        self.writer.display_terminal(len(self.valid_data_loader), epoch, mode='val', summary=True)


