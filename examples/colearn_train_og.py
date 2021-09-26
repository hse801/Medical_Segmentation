import time
from time import time
import os
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import lib.medloaders as medical_loaders
import lib.medzoo as medzoo
import lib.train as train
from lib.medzoo.CoLearning import YNet, init
from lib.losses3D.basic import compute_per_channel_dice
from lib.losses3D.colearn_loss import Co_DiceLoss, compute_dice
from lib.medzoo.CoLearning import net

if not os.path.exists(r'E:\HSE\Medical_Segmentation\saved_models\Colearn_checkpoints'):
    os.makedirs(r'E:\HSE\Medical_Segmentation\saved_models\Colearn_checkpoints')

cudnn.benchmark = True
Epoch = 1000
leaing_rate_base = 1e-5

batch_size = 2
num_workers = 4
pin_memory = True


def main():
    net = YNet(training=True)
    net.apply(init)
    net = torch.nn.DataParallel(net).cuda()
    print(net)

    # Define dataloader
    # train_dl = DataLoader(train_fix_ds, batch_size, True, num_workers=num_workers, pin_memory=pin_memory)
    train_dl, _, _ = medical_loaders.colearn_dataloader.generate_lung_dataset()

    loss_func = Co_DiceLoss()
    opt = torch.optim.Adam(net.parameters(), lr=leaing_rate_base)
    lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, [400])

    # start = time()

    for epoch in range(Epoch):
        print(f'Epoch {epoch} training ...')
        loss_step = []
        sum_dice = 0

        for step, (ct, seg, pet) in enumerate(train_dl):

            ct = ct.cuda()
            seg = seg.cuda()
            pet = pet.cuda()

            if torch.numel(seg[seg > 0]) == 0:
                continue

            ct = Variable(ct)
            seg = Variable(seg)
            pet = Variable(pet)
            # print(f'ct shape = {ct.shape}, pet shape = {pet.shape}, seg shape = {seg.shape}')
            outputs, outputs_temp = net(ct, pet)
            loss = loss_func(outputs, outputs_temp, seg)
            # print(f'outputs = {max(outputs)}')
            # print(f'seg = {max(seg)}')
            dice_score = compute_dice(outputs, seg)

            opt.zero_grad()
            loss.backward()
            opt.step()

            # xzl
            loss_step.append(loss.item())
            sum_dice += dice_score

            print(f'Step {step} : loss = {loss}, dice = {dice_score}')

        lr_decay.step()

        loss_mean = (sum(loss_step)/len(loss_step))
        dice_mean = sum_dice / len(loss_step)

        print('-------------------------------------------------------')
        print(f'Epoch: {epoch}, Loss mean = {loss_mean}, Dice mean = {dice_mean}')

            # if step % 20 is 0:
            #     print('epoch:{}, step:{}, loss:{:.3f}, time:{:.3f} min'
            #           .format(epoch, step, loss.item(), (time() - start) / 60))

        # now = time.localtime()

        # if epoch >= 150:
        torch.save(net.state_dict(),'E:\HSE\Medical_Segmentation\saved_models\Colearn_checkpoints\CoLearn_Epoch{}.pth'.format(epoch))


if __name__ == '__main__':
    main()
