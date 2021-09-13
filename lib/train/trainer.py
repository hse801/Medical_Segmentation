import numpy as np
import torch
from torch import nn as nn
from lib.utils.general import prepare_input
from lib.visual3D_temp.BaseWriter import TensorboardWriter
from lib.losses3D import *


class Trainer:
    """
    Trainer class
    """

    def __init__(self, args, model, criterion, optimizer, dice_score, train_data_loader,
                 valid_data_loader=None):

        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dice_score = dice_score
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

            input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
            input_tensor.requires_grad = True
            # print(f'trainer.py: input_tensor size = {input_tensor.size()}')
            output = self.model(input_tensor)

            # print(f'trainer.py: output dim = {output.size()}, target.dim = {target.size()}')
            # loss_dice = self.criterion(output, target)
            # print(f'dice loss = {loss_dice}')

            # L = WeightedCrossEntropyLoss()
            # loss = L(output, target)
            loss, per_ch_score = self.criterion(output, target)
            # loss = self.criterion(output, target)

            # loss = WeightedCrossEntropyLoss()
            loss.backward()
            # print(f'l2 loss = {loss_dice}')
            self.optimizer.step()
            # self.lr_scheduler.step()

            self.writer.update_scores(batch_idx, loss.item(), per_ch_score, 'train',
                                      epoch * self.len_epoch + batch_idx)
            # self.writer.update_scores(batch_idx, loss_dice.item(), loss_dice.item(), 'train',
            #                           epoch * self.len_epoch + batch_idx)

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

                # _, per_ch_score = self.criterion(output, target)
                L = WeightedCrossEntropyLoss()
                loss = L(output, target)
                # loss = WeightedCrossEntropyLoss()
                _, per_ch_score = self.criterion(output, target)
                # loss = self.criterion(output, target)

                self.writer.update_scores(batch_idx, loss.item(), per_ch_score, 'val',
                                          epoch * self.len_epoch + batch_idx)
                # self.writer.update_scores(batch_idx, loss.item(), loss.item(), 'val',
                #                           epoch * self.len_epoch + batch_idx)

        self.writer.display_terminal(len(self.valid_data_loader), epoch, mode='val', summary=True)


