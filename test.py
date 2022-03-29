import os
import numpy as np
from tqdm import tqdm

import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader

import models
from data.datasets import Getdata
import cv2

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-gpu', '--gpu', default='0', type=str, 
                    help='Supprot one GPU & multiple GPUs.')
parser.add_argument('-cross_num', '--cross_num', default=1, type=int,
                    help='Cross Valid Set')
parser.add_argument('-epoch', '--epoch', default=300, type=int,
                    help='Epoch')
parser.add_argument('-workers', '--workers', default=0, type=int,
                    help='Workers')

## parse arguments
args = parser.parse_args()

gpu = args.gpu
Network_G = getattr(models, 'UnetGenerator')
workers = args.workers
resume = f'./ckpts/model_epoch_{args.epoch}.pt'
valid_txt = f"./data/heart/valid/valid_{args.cross_num}.txt"
print(f'Cross Set Number : {args.cross_num}')

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    device = (f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists('./test'):
        os.makedirs('./test')

    model_G = Network_G(in_channels = 1, out_channels = 1).to(device)
    # model_G.load_state_dict(torch.load(resume))

    valid_set = Getdata(valid_txt, 'valid')

    valid_loader = DataLoader(
            dataset=valid_set,
            batch_size=1,
            shuffle = False,
            num_workers=workers)
    
    model_G.eval()
    
    for i, data in tqdm(enumerate(valid_loader)):
        real_A = data[0].to(device)
        real_B = data[1].to(device)
        
        fake_B = np.squeeze(model_G(real_A).to('cpu').detach().numpy())
        
        real_A = np.squeeze(real_A.to('cpu').detach().numpy())
        real_B = np.squeeze(real_B.to('cpu').detach().numpy())
        
        name = valid_set.mat_ffs[i].split("/")[-1].split(".mat")[0]
        cv2.imwrite(f"./test/{name}.jpg",real_A*255)
        cv2.imwrite(f"./test/{name}_real.jpg",real_B*255)
        cv2.imwrite(f"./test/{name}_fake.jpg",fake_B*255)

            

if __name__ == '__main__':
    main()





