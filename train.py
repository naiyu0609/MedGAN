import os
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision

import models
from data.datasets import Getdata
import cv2

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-gpu', '--gpu', default='0', type=str, 
                    help='Supprot one GPU & multiple GPUs.')
parser.add_argument('-batch_size', '--batch_size', default=1, type=int,
                    help='Batch size')
parser.add_argument('-epoch', '--epoch', default=5000, type=int,
                    help='Epoch')
parser.add_argument('-save_frq', '--save_frq', default=100, type=int,
                    help='Save frequency')
parser.add_argument('-cross_num', '--cross_num', default=1, type=int,
                    help='Cross Valid Set')
parser.add_argument('-workers', '--workers', default=0, type=int,
                    help='Workers')

## parse arguments
args = parser.parse_args()

gpu = args.gpu
batch_size = args.batch_size
eps = 1e-5
seed = 1024
num_trainG = 3
num_epochs = args.epoch
save_frq = args.save_frq
criterionGAN = nn.MSELoss()
criterionL1 = nn.L1Loss()
real_label=torch.tensor(1)
fake_label=torch.tensor(0)
lambda_L1 = 100.0
Network_G = getattr(models, 'UnetGenerator')
Network_D = getattr(models, 'NLayerDiscriminator')
lr = 1e-4
weight_decay = 1e-5
amsgrad = True
workers = args.workers
train_txt = f"./data/heart/train/train_{args.cross_num}.txt"
valid_txt = f"./data/heart/valid/valid_{args.cross_num}.txt"


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    device = (f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if not os.path.exists('./ckpts'):
        os.makedirs('./ckpts')
    if not os.path.exists('./picture'):
        os.makedirs('./picture')

    model_G = Network_G(in_channels = 1, out_channels = 1).to(device)
    model_D = Network_D(input_nc = 2).to(device)
    model_VGG = torchvision.models.vgg19(pretrained=True).to(device)
    optimizer_G = torch.optim.Adam(model_G.parameters(), lr = lr, weight_decay= weight_decay, amsgrad = amsgrad)
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr = lr, weight_decay= weight_decay, amsgrad = amsgrad)

    print(f'Cross Set Number : {args.cross_num}')
    print("-------------- New training session ----------------")

    train_set = Getdata(train_txt, 'train')
    valid_set = Getdata(valid_txt, 'valid')

    train_loader = DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle = True,
            num_workers=workers)

    valid_loader = DataLoader(
            dataset=valid_set,
            batch_size=1,
            shuffle = False,
            num_workers=workers)

    tGloss_plt = []
    tDloss_plt = []
    tGlosses = AverageMeter()
    tDlosses = AverageMeter()
    
    vGloss_plt = []
    vDloss_plt = []
    vGlosses = AverageMeter()
    vDlosses = AverageMeter()

    for j in range(num_epochs):
        print(f"Epoch:{j+1}/{num_epochs}")
        print('Train')
        model_G.train()
        model_D.train()
        for data in tqdm(train_loader):
            adjust_learning_rate(optimizer_G, j+1, num_epochs, lr)
            adjust_learning_rate(optimizer_D, j+1, num_epochs, lr)

            real_A = data[0].to(device)
            real_B = data[1].to(device)
            
            # First, G(A) should fake the discriminator
            for t in range(num_trainG):
                fake_B = model_G(real_A)
                
                optimizer_G.zero_grad()
                # First, G(A) should fake the discriminator
                fake_AB = torch.cat((real_A, fake_B), 1)
                pred_fake = model_D(fake_AB)
                target_tensor = real_label
                target_tensor = target_tensor.expand_as(pred_fake).to(device)
                loss_G_GAN = criterionGAN(pred_fake.to(torch.float32), target_tensor.to(torch.float32))
                # Second, G(A) = B
                loss_G_L1 = criterionL1(fake_B, real_B) * lambda_L1
                # Third, Feature(G(A)) = Feature(B)
                loss_G_feature = criterionGAN(model_VGG(torch.cat([fake_B, fake_B, fake_B], dim=1)), model_VGG(torch.cat([real_B, real_B, real_B], dim=1)))
                
                # combine loss and calculate gradients
                loss_G = loss_G_GAN + loss_G_L1 + loss_G_feature
                loss_G.backward()
                optimizer_G.step()
                
            
            optimizer_D.zero_grad()
            # Fake; stop backprop to the generator by detaching fake_B
            fake_AB = torch.cat((real_A, fake_B), 1) 
            pred_fake = model_D(fake_AB.detach())
            target_tensor = fake_label
            target_tensor = target_tensor.expand_as(pred_fake).to(device)
            loss_D_fake = criterionGAN(pred_fake.to(torch.float32), target_tensor.to(torch.float32))
            # Real
            real_AB = torch.cat((real_A, real_B), 1)
            pred_real = model_D(real_AB)
            target_tensor = real_label
            target_tensor = target_tensor.expand_as(pred_real).to(device)
            loss_D_real = criterionGAN(pred_real.to(torch.float32), target_tensor.to(torch.float32))

            loss_D = (loss_D_fake + loss_D_real) * 0.5
            loss_D.backward()
            optimizer_D.step()
            
            tGlosses.update(loss_G.item(), batch_size)
            tDlosses.update(loss_D.item(), batch_size)

        print()
        print('-----------------------Train-----------------------')
        print('G Loss {:.7f}'.format(tGlosses.avg))
        print('D Loss {:.7f}'.format(tDlosses.avg))
        print('---------------------------------------------------')

        tGloss_plt.append(tGlosses.avg)
        tDloss_plt.append(tDlosses.avg)
        tGlosses.reset()
        tDlosses.reset()


        # if ((j+1) % 5 == 0 ) & (j != 1):
        print('Vaild')
        model_G.eval()
        model_D.eval()
        for i, data in enumerate(tqdm(valid_loader)):
            real_A = data[0].to(device)
            real_B = data[1].to(device)
            
            fake_B = model_G(real_A)
                
            # First, G(A) should fake the discriminator
            fake_AB = torch.cat((real_A, fake_B), 1)
            pred_fake = model_D(fake_AB)
            target_tensor = real_label
            target_tensor = target_tensor.expand_as(pred_fake).to(device)
            loss_G_GAN = criterionGAN(pred_fake.to(torch.float32), target_tensor.to(torch.float32))
            # Second, G(A) = B
            loss_G_L1 = criterionL1(fake_B, real_B) * lambda_L1
            # Third, Feature(G(A)) = Feature(B)
            loss_G_feature = criterionGAN(model_VGG(torch.cat([fake_B, fake_B, fake_B], dim=1)), model_VGG(torch.cat([real_B, real_B, real_B], dim=1)))
            
            # combine loss and calculate gradients
            loss_G = loss_G_GAN + loss_G_L1 + loss_G_feature
            
            # Fake; stop backprop to the generator by detaching fake_B
            fake_AB = torch.cat((real_A, fake_B), 1) 
            pred_fake = model_D(fake_AB.detach())
            target_tensor = fake_label
            target_tensor = target_tensor.expand_as(pred_fake).to(device)
            loss_D_fake = criterionGAN(pred_fake.to(torch.float32), target_tensor.to(torch.float32))
            # Real
            real_AB = torch.cat((real_A, real_B), 1)
            pred_real = model_D(real_AB)
            target_tensor = real_label
            target_tensor = target_tensor.expand_as(pred_real).to(device)
            loss_D_real = criterionGAN(pred_real.to(torch.float32), target_tensor.to(torch.float32))

            loss_D = (loss_D_fake + loss_D_real) * 0.5
            

            if (j+1) % (save_frq/2) == 0 :
                name = valid_set.mat_ffs[i].split("/")[-1].split(".mat")[0]
                if not os.path.exists(f'./picture_{j+1}'):
                    os.makedirs(f'./picture_{j+1}')
                cv2.imwrite(f"./picture_{j+1}/{name}_G.jpg",np.squeeze(fake_B.to('cpu').detach().numpy())*255)
                cv2.imwrite(f"./picture_{j+1}/{name}_real.jpg",np.squeeze(real_B.to('cpu').detach().numpy())*255)
            

            vGlosses.update(loss_G.item(), batch_size)
            vDlosses.update(loss_D.item(), batch_size)

        print()
        print('-----------------------Valid-----------------------')
        print('G Loss {:.7f}'.format(vGlosses.avg))
        print('D Loss {:.7f}'.format(vDlosses.avg))
        print('---------------------------------------------------')

        vGloss_plt.append(vGlosses.avg)
        vDloss_plt.append(vDlosses.avg)
        vGlosses.reset()
        vDlosses.reset()

        print()

        if ((j+1) % save_frq == 0 ) & (j != 1):
            plot((j+1), tGloss_plt, tDloss_plt, vGloss_plt, vDloss_plt)
            file_name = os.path.join(f'./ckpts/modelG_epoch_{j+1}.pt')
            torch.save(model_G.state_dict(),file_name)
            file_name = os.path.join(f'./ckpts/modelD_epoch_{j+1}.pt')
            torch.save(model_D.state_dict(),file_name)
            


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val_all = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val_all.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.std = np.std(np.array(self.val_all))


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)

def plot(epoch, tGloss_plt, tDloss_plt, vGloss_plt, vDloss_plt):
    plt.figure()
    plt.plot(tGloss_plt,'-', label='Train')
    plt.plot(vGloss_plt,'-', label='Valid')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('G Train vs Valid Loss')
 
    plt.savefig(f'./picture/epoch{epoch} G Train vs Valid Loss.png')  

    plt.figure()
    plt.plot(tDloss_plt,'-', label='Train')
    plt.plot(vDloss_plt,'-', label='Valid')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('D Train vs Valid Loss')
 
    plt.savefig(f'./picture/epoch{epoch} D Train vs Valid Loss.png')  

    plt.close('all')

if __name__ == '__main__':
    main()





