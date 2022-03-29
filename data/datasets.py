from scipy.io import loadmat
from torchvision import transforms
import os
import numpy as np
import torch


class Getdata():
    def __init__(self, list_file, mode = 'train', root = '../heart/T1 data/KVSGH_TOF/syn_raw_mat_gan/'):
        paths = []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                path = os.path.join(root, line + '.mat')
                paths.append(path)

        self.mat_ffs = paths
        
        if mode == 'train':
            self.transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224,224)),
                transforms.ToTensor()
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor()
            ])
        
    def __len__(self):
        return len(self.mat_ffs)    
    
    def __getitem__(self, idx):
        mat = loadmat(self.mat_ffs[idx])
        
        T1map = mat['T1map']/np.max(mat['T1map'])
        syn_im_gan = mat['syn_im_gan']/np.max(mat['syn_im_gan'])
        
        T1map = T1map.astype(np.float32)
        syn_im_gan = syn_im_gan.astype(np.float32)
        
        T1map = np.stack([T1map])
        syn_im_gan = np.stack([syn_im_gan])
        
        T1map = self.transforms(torch.from_numpy(T1map))
        syn_im_gan = self.transforms(torch.from_numpy(syn_im_gan))

        return T1map, syn_im_gan

