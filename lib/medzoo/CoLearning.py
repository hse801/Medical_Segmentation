import torch
import torch.nn as nn
import torch.nn.functional as F


class YNet(nn.Module):
    def __init__(self, training):
        super().__init__()

        self.training = training

        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(1, 8, 3, 1, padding=1),
            nn.PReLU(8),
        )
        self.encoder_stage11 = nn.Sequential(
            nn.Conv3d(1, 8, 3, 1, padding=1),
            nn.PReLU(8),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(16, 16, 3, 1, padding=1),
            nn.PReLU(16),

            nn.Conv3d(16, 16, 3, 1, padding=1),
            nn.PReLU(16),
        )
        self.encoder_stage22 = nn.Sequential(
            nn.Conv3d(16, 16, 3, 1, padding=1),
            nn.PReLU(16),

            nn.Conv3d(16, 16, 3, 1, padding=1),
            nn.PReLU(16),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )
        self.encoder_stage33 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
        )
        self.encoder_stage44 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
        )

        self.encoder_stage5 = nn.Sequential(

            nn.Conv3d(128 + 128, 128, 1, 1, padding=0),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 1, 1, padding=0),
            nn.PReLU(128),

        )
        self.encoder_stage55 = nn.Sequential(

            nn.Conv3d(128 + 128, 128, 1, 1, padding=0),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 1, 1, padding=0),
            nn.PReLU(128),
        )

        self.decoder_stage1 = nn.Sequential(

            nn.Conv3d(128 + 64 + 64 + 128, 128, 1, 1, padding=0),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 1, 1, padding=0),
            nn.PReLU(128),
        )
        self.decoder_stage11 = nn.Sequential(

            nn.Conv3d(128 + 64 + 64, 128, 1, 1, padding=0),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 1, 1, padding=0),
            nn.PReLU(128),

        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(64 + 32 + 32 + 32 + 64, 64, 1, 1, padding=0),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 1, 1, padding=0),
            nn.PReLU(64),
        )
        self.decoder_stage22 = nn.Sequential(
            nn.Conv3d(64 + 32 + 32, 64, 1, 1, padding=0),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 1, 1, padding=0),
            nn.PReLU(64),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(32 + 16 + 32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )
        self.decoder_stage33 = nn.Sequential(
            nn.Conv3d(32 + 16, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(16 + 8, 16, 3, 1, padding=1),
            nn.PReLU(16),
        )
        self.decoder_stage44 = nn.Sequential(
            nn.Conv3d(16 + 8, 16, 3, 1, padding=1),
            nn.PReLU(16),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(8, 16, 2, 2),
            nn.PReLU(16),

            nn.Conv3d(16, 16, 1, 1),
            nn.PReLU(16),

            nn.Conv3d(16, 16, 1, 1),
            nn.PReLU(16),
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(16, 32, 2, 2),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 1, 1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 1, 1),
            nn.PReLU(32),
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(32, 64, 2, 2),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 1, 1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 1, 1),
            nn.PReLU(64),
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv3d(64, 128, 2, 2),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 1, 1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 1, 1),
            nn.PReLU(128),
        )

        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose3d(128, 128, 2, 2),
            nn.PReLU(128)
        )
        self.up_conv11 = nn.Sequential(
            nn.ConvTranspose3d(128, 128, 2, 2),
            nn.PReLU(128)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.PReLU(64)
        )
        self.up_conv22 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.PReLU(64)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.PReLU(32)
        )
        self.up_conv33 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.PReLU(32)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 2, 2),
            nn.PReLU(16)
        )
        self.up_conv44 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 2, 2),
            nn.PReLU(16)
        )

        self.map = nn.Sequential(
            nn.Conv3d(16, 1, 1, 1),
            nn.Sigmoid()
        )
        self.map_temp = nn.Sequential(
            nn.Conv3d(16, 1, 1, 1),
            nn.Sigmoid()
        )

        self.FFB_1 = nn.Sequential(
            nn.Conv3d(128 + 128, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),
        )

        self.DUC_1_1 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2, groups=4),
            nn.PReLU(64),
        )
        self.DUC_1_2 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2, groups=4),
            nn.PReLU(32),
        )

        self.FFB_2 = nn.Sequential(
            nn.Conv3d(64 + 64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
        )
        self.DUC_2_1 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2, groups=4),
            nn.PReLU(32),
        )

        self.FFB_3 = nn.Sequential(
            nn.Conv3d(32 + 32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.final_activation = nn.Sigmoid()

    def forward(self, inputs, inputs_pet):
        """
        [Batch, Channel, Depth, Width, Height]
        inputs = torch.Size([2, 1, 80, 128, 160]), inputs_pet = torch.Size([2, 1, 80, 128, 160])
        long_range1 = torch.Size([2, 8, 80, 128, 160]), long_range1_pet = torch.Size([2, 8, 80, 128, 160])
        short_range1 = torch.Size([2, 16, 40, 64, 80]), short_range1_pet = torch.Size([2, 16, 40, 64, 80])
        long_range2 = torch.Size([2, 16, 40, 64, 80]), long_range2_pet = torch.Size([2, 16, 40, 64, 80])
        short_range2 = torch.Size([2, 32, 20, 32, 40]), short_range2_pet = torch.Size([2, 32, 20, 32, 40])
        long_range3 = torch.Size([2, 32, 20, 32, 40]), long_range3_pet = torch.Size([2, 32, 20, 32, 40])
        short_range3 = torch.Size([2, 64, 10, 16, 20]), short_range3_pet = torch.Size([2, 64, 10, 16, 20])
        long_range4 = torch.Size([2, 64, 10, 16, 20]), long_range4_pet = torch.Size([2, 64, 10, 16, 20])
        short_rang4 = torch.Size([2, 128, 5, 8, 10]), short_range4_pet = torch.Size([2, 128, 5, 8, 10])
        FFB_out_1 = torch.Size([2, 128, 5, 8, 10]), DUC_1_1_out = torch.Size([2, 64, 10, 16, 20]), DUC_1_2_out = torch.Size([2, 32, 20, 32, 40])

        :param inputs: ct array
        :param inputs_pet: pet array
        :return:
        """
        print(f'inputs = {inputs.size()}, inputs_pet = {inputs_pet.size()}')
        long_range1 = self.encoder_stage1(inputs) + inputs
        long_range1_pet = self.encoder_stage11(inputs_pet)

        short_range1 = self.down_conv1(long_range1)
        short_range1_pet = self.down_conv1(long_range1_pet)
        print(f'long_range1 = {long_range1.size()}, long_range1_pet = {long_range1_pet.size()}')
        print(f'short_range1 = {short_range1.size()}, short_range1_pet = {short_range1_pet.size()}')

        long_range2 = self.encoder_stage2(short_range1) + short_range1
        long_range2 = F.dropout(long_range2, 0.5, self.training)
        long_range2_pet = self.encoder_stage22(short_range1_pet)
        long_range2_pet = F.dropout(long_range2_pet, 0.5, self.training)

        short_range2 = self.down_conv2(long_range2)
        short_range2_pet = self.down_conv2(long_range2_pet)
        print(f'long_range2 = {long_range2.size()}, long_range2_pet = {long_range2_pet.size()}')
        print(f'short_range2 = {short_range2.size()}, short_range2_pet = {short_range2_pet.size()}')

        long_range3 = self.encoder_stage3(short_range2) + short_range2
        long_range3 = F.dropout(long_range3, 0.5, self.training)
        long_range3_pet = self.encoder_stage33(short_range2_pet)
        long_range3_pet = F.dropout(long_range3_pet, 0.5, self.training)

        short_range3 = self.down_conv3(long_range3)
        short_range3_pet = self.down_conv3(long_range3_pet)

        print(f'long_range3 = {long_range3.size()}, long_range3_pet = {long_range3_pet.size()}')
        print(f'short_range3 = {short_range3.size()}, short_range3_pet = {short_range3_pet.size()}')


        long_range4 = self.encoder_stage4(short_range3) + short_range3
        long_range4 = F.dropout(long_range4, 0.5, self.training)
        long_range4_pet = self.encoder_stage44(short_range3_pet) + short_range3_pet
        long_range4_pet = F.dropout(long_range4_pet, 0.5, self.training)

        short_range4 = self.down_conv4(long_range4)
        short_range4_pet = self.down_conv4(long_range4_pet)

        print(f'long_range4 = {long_range4.size()}, long_range4_pet = {long_range4_pet.size()}')
        print(f'short_rang4 = {short_range4.size()}, short_range4_pet = {short_range4_pet.size()}')

        FFB_out_1 = self.FFB_1(torch.cat([short_range4, short_range4_pet], dim=1))
        FFB_out_1 = F.dropout(FFB_out_1, 0.5, self.training)
        # print('FFB_out_1.shape:', FFB_out_1.shape)                                        # [1, 256, 3, 16, 16]
        DUC_1_1_out = self.DUC_1_1(FFB_out_1)
        # print('DUC_1_1.shape:', DUC_1_1_out.shape)                                        # [1, 128, 6, 32, 32]
        DUC_1_2_out = self.DUC_1_2(DUC_1_1_out)
        # print('DUC_1_2.shape:', DUC_1_2_out.shape)                                        # [1, 64, 12, 64, 64]
        print(f'FFB_out_1 = {FFB_out_1.size()}, DUC_1_1_out = {DUC_1_1_out.size()}, DUC_1_2_out = {DUC_1_2_out.size()}')

        outputs_0 = self.encoder_stage55(torch.cat([short_range4, short_range4_pet], dim=1))+ short_range4 + short_range4_pet
        outputs_0 = F.dropout(outputs_0, 0.5, self.training)
        outputs = self.encoder_stage5(torch.cat([FFB_out_1, outputs_0],dim=1))
        outputs = F.dropout(outputs, 0.5, self.training)

        short_range5 = self.up_conv1(outputs)
        short_range55 = self.up_conv11(outputs_0)

        FFB_out_2 = self.FFB_2(torch.cat([long_range4, long_range4_pet], dim=1))
        FFB_out_2 = F.dropout(FFB_out_2, 0.5, self.training)
        # print('FFB_out_2.shape:', FFB_out_2.shape)                                          # [1, 128, 6, 32, 32]
        DUC_2_1_out = self.DUC_2_1(FFB_out_2)
        # print('DUC_2_1_out.shape:', DUC_2_1_out.shape)                                      # [1, 64, 12, 64, 64]

        outputs_1 = self.decoder_stage11(torch.cat([short_range55, long_range4, long_range4_pet], dim=1)) + short_range55
        outputs_1 = F.dropout(outputs_1, 0.5, self.training)
        outputs = self.decoder_stage1(torch.cat([short_range5, DUC_1_1_out, FFB_out_2, outputs_1], dim=1)) + short_range5
        outputs = F.dropout(outputs, 0.5, self.training)

        short_range6 = self.up_conv2(outputs)
        short_range66 = self.up_conv22(outputs_1)

        FFB_out_3 = self.FFB_3(torch.cat([long_range3, long_range3_pet], dim=1))
        FFB_out_3 = F.dropout(FFB_out_3, 0.5, self.training)
        # print('FFB_out_3.shape:', FFB_out_3.shape)                                           # [1, 64, 12, 64, 64]

        outputs_2 = self.decoder_stage22(torch.cat([short_range66, long_range3, long_range3_pet], dim=1)) + short_range66
        outputs_2 = F.dropout(outputs_2, 0.5, self.training)
        outputs = self.decoder_stage2(torch.cat([short_range6, DUC_1_2_out, DUC_2_1_out, FFB_out_3, outputs_2], dim=1)) + short_range6
        outputs = F.dropout(outputs, 0.5, self.training)

        short_range7 = self.up_conv3(outputs)
        short_range77 = self.up_conv33(outputs_2)

        outputs_3 = self.decoder_stage33(torch.cat([short_range77, long_range2], dim=1)) + short_range77
        outputs_3 = F.dropout(outputs_3, 0.5, self.training)
        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2, outputs_3], dim=1)) + short_range7
        outputs = F.dropout(outputs, 0.5, self.training)

        short_range8 = self.up_conv4(outputs)
        short_range88 = self.up_conv44(outputs_3)

        outputs_4 = self.decoder_stage44(torch.cat([short_range88, long_range1], dim=1)) + short_range88
        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8

        outputs = self.map(outputs)
        outputs_temp = self.map_temp(outputs_4)

        outputs = self.final_activation(outputs)
        outputs_temp = self.final_activation(outputs_temp)

        return outputs, outputs_temp


def init(module):
    if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
        nn.init.kaiming_normal_(module.weight.data, 0.25)
        nn.init.constant_(module.bias.data, 0)


net = YNet(training=True)
net.apply(init)