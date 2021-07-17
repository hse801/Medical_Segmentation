import torch.optim as optim
import torch

from .COVIDNet import CovidNet, CNN
from .DenseVoxelNet import DenseVoxelNet
from .Densenet3D import DualPathDenseNet, DualSingleDenseNet, SinglePathDenseNet
from .HighResNet3D import HighResNet3D
from .HyperDensenet import HyperDenseNet, HyperDenseNet_2Mod
from .ResNet3DMedNet import generate_resnet3d
from .ResNet3D_VAE import ResNet3dVAE
from .SkipDenseNet3D import SkipDenseNet3D
from .Unet2D import Unet
from .Unet3D import UNet3D
from .Vnet import VNet, VNetLight
from .TransBTS.TransBTS_downsample8x_skipconnection import TRANSBTS
from .Attention_Networks.UNet_multi_att_3D import AttentionUNet
from .Unet3D_OG.Unet3D_og import Abstract3DUNet, UNet3DOG, ResidualUNet3D

model_list = ['UNET3D', 'DENSENET1', "UNET2D", 'DENSENET2', 'DENSENET3', 'HYPERDENSENET', "SKIPDENSENET3D",
              "DENSEVOXELNET", 'VNET', 'VNET2', "RESNET3DVAE", "RESNETMED3D", "COVIDNET1", "COVIDNET2", "CNN",
              "HIGHRESNET", "TRANSBTS", "ATTENTIONUNET", "UNET3DOG", "RESUNETOG"]


def create_model(args):
    model_name = args.model
    assert model_name in model_list
    optimizer_name = args.opt
    lr_scheduler_name = args.lrscheduler
    lr = args.lr
    in_channels = args.inChannels
    num_classes = args.classes
    weight_decay = 0.0000000001
    print("Building Model . . . . . . . ." + model_name)

    if model_name == 'VNET2':
        model = VNetLight(in_channels=in_channels, elu=False, classes=num_classes)
    elif model_name == 'VNET':
        model = VNet(in_channels=in_channels, elu=False, classes=num_classes)
    elif model_name == 'UNET3D':
        model = UNet3D(in_channels=in_channels, n_classes=num_classes, base_n_filter=24)
    elif model_name == 'ATTENTIONUNET':
        model = AttentionUNet(n_classes=num_classes, in_channels=in_channels)
    elif model_name == 'UNET3DOG':
        model = UNet3DOG(in_channels=1, out_channels=1)
    elif model_name == 'RESUNETOG':
        model = ResidualUNet3D(in_channels=1, out_channels=1)
    elif model_name == 'TRANSBTS':
        model = TRANSBTS(_conv_repr=True, _pe_type="learned")
    elif model_name == 'DENSENET1':
        model = SinglePathDenseNet(in_channels=in_channels, classes=num_classes)
    elif model_name == 'DENSENET2': # CHANNEL 2 OR 3
        model = DualPathDenseNet(in_channels=in_channels, classes=num_classes)
    elif model_name == 'DENSENET3': # CHANNEL 2 OR 3
        model = DualSingleDenseNet(in_channels=in_channels, drop_rate=0.1, classes=num_classes)
    elif model_name == "UNET2D":
        model = Unet(in_channels, num_classes)
    elif model_name == "RESNET3DVAE":
        model = ResNet3dVAE(in_channels=in_channels, classes=num_classes, dim=args.dim)
    elif model_name == "SKIPDENSENET3D":
        model = SkipDenseNet3D(growth_rate=16, num_init_features=32, drop_rate=0.1, classes=num_classes)
    elif model_name == "COVIDNET1": # 2D
        model = CovidNet('small', num_classes)
    elif model_name == "COVIDNET2": # 2D
        model = CovidNet('large', num_classes)
    elif model_name == "CNN":
        model = CNN(num_classes, 'resnet18')
    elif model_name == "HYPERDENSENET":
        if in_channels == 2:
            model = HyperDenseNet_2Mod(classes=num_classes)
        elif in_channels == 3:
            model = HyperDenseNet(classes=num_classes)
        else:
            raise NotImplementedError
    elif model_name == "DENSEVOXELNET":
        model = DenseVoxelNet(in_channels=in_channels, classes=num_classes)
    elif model_name == "HIGHRESNET":
        model = HighResNet3D(in_channels=in_channels, classes=num_classes)
    elif model_name == "RESNETMED3D":
        depth = 18
        model = generate_resnet3d(in_channels=in_channels, classes=num_classes, model_depth=depth)

    print(model_name, 'Number of params: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay) # 0.5 before
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    # if lr_scheduler_name == 'lambdalr':
    #     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.85 ** epoch)
    # elif lr_scheduler_name == 'steplr':
    #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    return model, optimizer