# Python libraries
import argparse
import os
import torchvision
import torchsummary
import torch
import os.path as osp
import timeit
import lib.medloaders as medical_loaders
import lib.medzoo as medzoo
import lib.train as train
# Lib files
import lib.utils as utils
from lib.losses3D import DiceLoss
from lib.losses3D import WeightedCrossEntropyLoss
from lib.losses3D import PixelWiseCrossEntropyLoss
from lib.losses3D import WeightedSmoothL1Loss

import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 1777777

local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    args = get_arguments()
    utils.reproducibility(args, seed)
    utils.make_dirs(args.save)

    train_generator, val_generator, _ = medical_loaders.thyroid_dataloader.generate_thyroid_dataset()
    # training_generator, val_generator, full_volume, affine = medical_loaders.generate_datasets(args,
    #                                                                                            path='.././datasets')
    model, optimizer = medzoo.create_model(args)
    criterion = DiceLoss(classes=args.classes)
    # criterion = WeightedSmoothL1Loss()
    # criterion = PixelWiseCrossEntropyLoss()
    # dice_score = DiceLoss(classes=args.classes)

    # load checkpoint...
    if args.resume:
        print('loading from checkpoint: {}'.format(args.reload_path))
        if os.path.exists(args.reload_path):
            model.load_state_dict(torch.load(args.reload_path, map_location=torch.device('cpu')))
        else:
            print('File not exists in the reload path: {}'.format(args.reload_path))

    if args.cuda:
        model = model.cuda()
        print(model)
        # torchsummary.summary(model, (1, 64, 64, 64))
        # net = torchvision.model()
        # print(net)

    start_time = time.time()
    trainer = train.Trainer_res(args, model, criterion, optimizer, train_data_loader=train_generator,
                            valid_data_loader=val_generator)
    trainer.training()

    end_time = time.time()
    total_time = (end_time-start_time)/3600
    print('The total training time is {:.2f} hours'.format(total_time))


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=2)
    parser.add_argument('--dataset_name', type=str, default="thyroid")
    parser.add_argument('--dim', nargs="+", type=int, default=(256, 256, 256))
    parser.add_argument('--nEpochs', type=int, default=100)
    parser.add_argument('--classes', type=int, default=2)
    parser.add_argument('--samples_train', type=int, default=100)
    parser.add_argument('--samples_val', type=int, default=100)
    parser.add_argument('--inChannels', type=int, default=1)
    parser.add_argument('--inModalities', type=int, default=1)
    parser.add_argument('--terminal_show_freq', default=50)
    parser.add_argument('--threshold', default=0.1, type=float)
    parser.add_argument('--augmentation', action='store_true', default=True)
    parser.add_argument('--normalization', default='full_volume_mean', type=str,
                        help='Tensor normalization: options ,max_min,',
                        choices=('max_min', 'full_volume_mean', 'brats', 'max', 'mean'))
    parser.add_argument('--split', default=0.8, type=float, help='Select percentage of training data(default: 0.8)')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--loadData', default=False)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument("--reload_path", type=str, default='snapshots/conresnet/ConResNet_40000.pth')
    parser.add_argument("--reload_from_checkpoint", type=str2bool, default=False)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--model', type=str, default='UNET3D',
                        choices=('VNET', 'VNET2', 'UNET3D', 'DENSENET1', 'DENSENET2', 'DENSENET3', 'HYPERDENSENET',
                                 'SKIPDENSENET3D', 'COVIDNET1', 'COVIDNET2', 'RESNETMED3D', 'HIGHRESNET',
                                 'TRANSBTS', 'RESNET3DVAE', 'DENSEVOXELNET', 'ATTENTIONUNET', 'UNET3DOG', 'RESUNETOG',
                                 'RESUNETKIDNEY', 'CONRESNET', 'DEEPMEDIC', 'RESUNETOGT', 'RESUNETOGL'))
    parser.add_argument('--opt', type=str, default='adam',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--lrscheduler', type=str, default='lambdalr',
                        choices=('lambdalr', 'steplr'))
    parser.add_argument('--log_dir', type=str,
                        default='../runs/')

    args = parser.parse_args()

    args.save = '../saved_models/' + args.model + '_checkpoints/' + args.model + '_{}_{}'.format(
        utils.datestr(), args.dataset_name)
    return args


if __name__ == '__main__':
    main()
