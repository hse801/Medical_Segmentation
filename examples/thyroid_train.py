# Python libraries
import argparse
import os
import torchvision
import torchsummary

import lib.medloaders as medical_loaders
import lib.medzoo as medzoo
import lib.train as train
# Lib files
import lib.utils as utils
from lib.losses3D import DiceLoss

import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 1777777

local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def main():
    args = get_arguments()
    utils.reproducibility(args, seed)
    utils.make_dirs(args.save)

    train_generator, val_generator, _ = medical_loaders.thyroid_dataloader.generate_thyroid_dataset()
    # training_generator, val_generator, full_volume, affine = medical_loaders.generate_datasets(args,
    #                                                                                            path='.././datasets')
    model, optimizer = medzoo.create_model(args)
    criterion = DiceLoss(classes=args.classes)

    if args.cuda:
        model = model.cuda()
        print(model)
        # torchsummary.summary(model, (1, 64, 64, 64))
        # net = torchvision.model()
        # print(net)

    start_time = time.time()
    trainer = train.Trainer(args, model, criterion, optimizer, train_data_loader=train_generator,
                            valid_data_loader=val_generator)
    trainer.training()

    end_time = time.time()
    total_time = (end_time-start_time)/3600
    print('The total training time is {:.2f} hours'.format(total_time))


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=2)
    parser.add_argument('--dataset_name', type=str, default="thyroid")
    parser.add_argument('--dim', nargs="+", type=int, default=(64, 64, 64))
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
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--model', type=str, default='UNET3D',
                        choices=('VNET', 'VNET2', 'UNET3D', 'DENSENET1', 'DENSENET2', 'DENSENET3', 'HYPERDENSENET',
                                 'SKIPDENSENET3D', 'COVIDNET1', 'COVIDNET2', 'RESNETMED3D', 'HIGHRESNET',
                                 'TRANSBTS', 'RESNET3DVAE', 'DENSEVOXELNET', 'ATTENTIONUNET', 'UNET3DOG', 'RESUNETOG',
                                 'RESUNETKIDNEY', 'CONRESNET'))
    parser.add_argument('--opt', type=str, default='adam',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--lrscheduler', type=str, default='lambdalr',
                        choices=('lambdalr', 'steplr'))
    parser.add_argument('--log_dir', type=str,
                        default='../runs/')

    args = parser.parse_args()

    args.save = '../saved_models/' + args.model + '_checkpoints/' + args.model + '_{}_{}_'.format(
        utils.datestr(), args.dataset_name)
    return args


if __name__ == '__main__':
    main()
