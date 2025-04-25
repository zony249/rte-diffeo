import os 
import sys 
from copy import deepcopy
from typing import List, Dict, Union, Tuple, Optional
from glob import glob

import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.color import rgb2gray
import torch.optim as optim
from skimage import io
from skimage.filters import gaussian
from skimage.transform import pyramid_gaussian
from torchvision.transforms.functional import gaussian_blur, resize, pad
from torch.nn.functional import interpolate
from skimage.transform import resize
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from torch.utils.data import Dataset, DataLoader
import struct
from array import array
import shutil

from resnet import ResNet

from homography import Trainer, Homography, mse, dv_loss, histogram_mutual_information


random_seed = 1
torch.backends.cudnn.enabled = True
torch.manual_seed(random_seed)

# Checking GPU availability
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)


I = io.imread("7.png").astype(np.float32)/255.0
J = io.imread("74.png").astype(np.float32)/255.0

I_ = torch.tensor(gaussian(I,1.0).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
J_ = torch.tensor(gaussian(J,1.0).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)

I_ = pad(I_, 2)
J_ = pad(J_, 2)

h_ = I_.shape[2]
w_ = I_.shape[3]

y_, x_ = np.meshgrid(np.arange(0,h_), np.arange(0,w_),indexing='ij')
y_, x_ = 2.0*y_/(h_-1) - 1.0, 2.0*x_/(w_-1) - 1.0

xy_ = torch.tensor(np.stack([x_,y_],2),dtype=torch.float32).unsqueeze(0).to(DEVICE)

print(h_, w_)



class Knee(Dataset): 
    def __init__(self, dir_path="./"): 
        super().__init__()
        k1_path = os.path.join(dir_path, "knee1.bmp")
        k2_path = os.path.join(dir_path, "knee2.bmp")
        self.knee1 = io.imread(k1_path).astype(np.float32) / 255.0
        self.knee2 = io.imread(k2_path).astype(np.float32) / 255.0

        if len(self.knee1.shape) == 2:
            self.knee1 = self.knee1[None, ...]
        if len(self.knee2.shape) == 2:
            self.knee2 = self.knee2[None, ...]

        self.mask = None

    def __len__(self): 
        return 1

    def __getitem__(self, idx): 

        h, w = 512, 512

        I_t = torch.from_numpy(self.knee1).unsqueeze(0).to(DEVICE)
        J_t = torch.from_numpy(self.knee2).unsqueeze(0).to(DEVICE)

        x = torch.linspace(-1, 1, w).to(DEVICE)
        y = torch.linspace(-1, 1, h).to(DEVICE)
        xy = torch.meshgrid(x, y, indexing="xy")
        xy = torch.stack(xy, dim=2).unsqueeze(0)

        I_t = F.grid_sample(I_t, xy, mode="bilinear", padding_mode="reflection", align_corners=False)
        J_t = F.grid_sample(J_t, xy, mode="bilinear", padding_mode="reflection", align_corners=False)

        return I_t, J_t, xy, None

class MNIST(Dataset): 
    def __init__(self, data_path="MNIST"): 
        super().__init__() 

        self.test_images_filepath = os.path.join(data_path, "t10k-images.idx3-ubyte") 
        self.test_labels_filepath = os.path.join(data_path, "t10k-labels.idx1-ubyte") 

        self.images, self.labels = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)

        self.labels = np.array(self.labels)
        self.images = np.array(self.images)


        self.images1 = [3, 2, 1, 18, 4, 23, 11, 0, 61, 12]
        self.images2 = [10, 5, 35, 30, 6, 15, 21, 17, 84, 9] 
        self.mask = None
    
    def read_images_labels(self, images_filepath, labels_filepath):       
        """
        taken from https://www.kaggle.com/code/hojjatk/read-mnist-dataset
        """ 
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def __len__(self): 
        return len(self.images1) 

    def __getitem__(self, idx): 
        
        image1 = self.images[self.images1[idx]]
        image2 = self.images[self.images2[idx]]

        image1 = np.pad(image1, [(2,2), (2, 2)])
        image2 = np.pad(image2, [(2,2), (2, 2)])

        h, w = image1.shape        

        x = torch.linspace(-1, 1, w).to(DEVICE)
        y = torch.linspace(-1, 1, h).to(DEVICE)
        xy = torch.meshgrid(x, y, indexing="xy")
        xy = torch.stack(xy, dim=2).unsqueeze(0)

        I_t = torch.from_numpy(image1).unsqueeze(0).unsqueeze(0).to(DEVICE) / 255.0
        J_t = torch.from_numpy(image2).unsqueeze(0).unsqueeze(0).to(DEVICE) / 255.0

        return I_t, J_t, xy, None


def train(dataset, output, hparams=None):


    if os.path.isdir(output): 
        shutil.rmtree(output)

    grids_path = os.path.join(output, "grids")
    flow_vec_path = os.path.join(output, "flow_vec")
    diff_path = os.path.join(output, "diff")
    results_path = os.path.join(output, "results")
    orig_path = os.path.join(output, "orig")
    samples_path = os.path.join(output, "samples")

    os.makedirs(grids_path)
    os.makedirs(flow_vec_path)
    os.makedirs(diff_path)
    os.makedirs(results_path)
    os.makedirs(orig_path)
    os.makedirs(samples_path)


    learning_rate = 1e-3

    for i, item in enumerate(dataset):
        if i < 0: 
            continue

        network = ResNet(num_layers=20).to(DEVICE)
        optimizer = optim.Adam(network.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        optimizer1 = optim.Adam(network.parameters(), lr=0.001)
        scaler = torch.amp.GradScaler()
        torch.cuda.empty_cache()

        I, J, xy, points = item

        h, w = I.shape[-2:]

        mask = dataset.mask


        # identity loss
        t0 = torch.zeros(1).to(DEVICE)
        for epoch in range(500):
            optimizer1.zero_grad()
            with torch.amp.autocast(DEVICE.type, dtype=torch.bfloat16):
                _, xyd, vs = network(J,xy)
                loss = torch.mean((xyd-xy)**2)
                loss = torch.mean((xyd-xy)**2)
            loss.backward()
            optimizer1.step()
            if epoch%100==0:
                print("Epoch:",epoch,"Id loss:","{:.10f}".format(loss.item()))

        del optimizer1
        torch.cuda.empty_cache() 

                # print("I_:", I_.shape)
                # print("J_:", J_.shape)
                # print("xy:", xy.shape)

        print("Epoch:",epoch,"Id loss:","{:.10f}".format(loss.item())
                )
        print("")
        # coarse to fine multi resolution optimization
        L = 10
        lam=hparams["lam"]
        nEpoch = 100
        batchsize = 1
        for epoch in range(nEpoch):

            res = np.sort((0.9)*np.random.rand(L-1)+0.1)
            res = np.concatenate((res,[1.0]))
            for level in range(L):
                optimizer.zero_grad()
                for step in range(batchsize):
                    loss,ss_loss = network.loss(I,J,xy)
                    total_loss = loss+lam*ss_loss
                    total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if epoch%1==0:
                    print("Epoch:",epoch,"Resolution:",L-1-level,"Total loss:","{:.6f}".format(total_loss.item()),
                                    "Image loss:","{:.6f}".format(loss.item()),
                                    "Constraint loss:","{:.10f}".format(ss_loss.item()),
                    )


            if epoch%1==0:
                print("")
            scheduler.step()

        for level in range(L):
            loss,ss_loss = network.loss(I,J,xy)
            total_loss = loss+lam*ss_loss
            print("Resolution:",L-1-level,"Epoch:",nEpoch,"Total loss:","{:.6f}".format(total_loss.item()),
                                "image loss:","{:.6f}".format(loss.item()),
                                "Constraint loss:","{:.10f}".format(ss_loss.item()),
            )


        # TEST 
        Jw, xyd, past = network(J,xy)
        # Jw = F.grid_sample(J_pre,xyd,padding_mode='reflection',align_corners=True)


        r1 = (J-I).max() - (J-I).min()
        r2 = (J-I).max() - (J-I).min()
        r3 = (Jw-I).max() - (Jw-I).min()

        m1 = (J-I).min()
        m2 = (J-I).min()
        m3 =  (Jw-I).min()
        
        max_range = torch.max(torch.tensor([r1, r2, r3])).detach().cpu().item() 
        min_val = torch.min(torch.tensor([m1, m2, m3])).detach().cpu().item()

        imgI = I[0].permute(1, 2, 0).detach().cpu().numpy() 
        imgJ = J[0].permute(1, 2, 0).detach().cpu().numpy() 
        imgJw = Jw[0].permute(1, 2, 0).detach().cpu().numpy() 

        mi_I_J = histogram_mutual_information(imgI, imgJ)
        mi_I_Jw = histogram_mutual_information(imgI, imgJw)



        

        fig=plt.figure(figsize=(10,4))

        fig.add_subplot(1,2,1)
        plt.title(f"J-I before Reg. MI: {mi_I_J:.5f}")
        plt.imshow((((J-I) - min_val)/max_range).squeeze(0).cpu().permute(1, 2, 0).data)

        fig.add_subplot(1,2,2)
        plt.imshow((((Jw-I) - min_val)/max_range).squeeze(0).cpu().permute(1, 2, 0).data)
        plt.title(f"J_deformed-I after Reg. MI: {mi_I_Jw:.5f}")
        plt.savefig(os.path.join(diff_path, f"{i}.png"), dpi=300)


        k = hparams["k"]
        fig=plt.figure(figsize=(5,5))
        d_ = (xyd - xy).squeeze()
        #fig=plt.figure(figsize=(5,5))
        fig.add_subplot(1,1,1)
        plt.quiver(d_.cpu().data[::k,::k,0], d_.cpu().data[::k,::k,1],color='r')
        plt.axis('equal')
        plt.title("Forward")

        plt.savefig(os.path.join(flow_vec_path, f"{i}.png"), dpi=300)







        fig=plt.figure(figsize=(14,4))

        fig.add_subplot(1,3,1)
        plt.title(f"I")
        plt.imshow(I.squeeze(0).cpu().permute(1, 2, 0).data)

        fig.add_subplot(1,3,2)
        plt.title(f"J")
        plt.imshow(J.squeeze(0).cpu().permute(1, 2, 0).data)

        fig.add_subplot(1,3,3)
        plt.imshow(Jw.squeeze(0).cpu().permute(1, 2, 0).data)
        plt.title(f"J_deformed")

        plt.savefig(os.path.join(results_path, f"{i}.png"), dpi=300)


        def plot_grid(x,y, ax=None, **kwargs):
            ax = ax or plt.gca()
            segs1 = np.stack((x,y), axis=2)
            segs2 = segs1.transpose(1,0,2)
            ax.add_collection(LineCollection(segs1, **kwargs))
            ax.add_collection(LineCollection(segs2, **kwargs))
            ax.autoscale()

        down_factor = hparams["down_factor"]
        h_resize = int(down_factor*h)
        w_resize = int(down_factor*w)

        grid_x = resize(xy.cpu()[:,:,:,0].squeeze().numpy(),(h_resize,w_resize))
        grid_y = resize(xy.cpu()[:,:,:,1].squeeze().numpy(),(h_resize,w_resize))
        distx = resize(xyd.cpu()[:,:,:,0].squeeze().detach().numpy(),(h_resize,w_resize))
        disty = resize(xyd.cpu()[:,:,:,1].squeeze().detach().numpy(),(h_resize,w_resize))

        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10, 4))
        plot_grid(grid_x,grid_y, ax=ax1,  color="lightgrey", linewidths=0.5)
        plot_grid(distx, disty, ax=ax1, color="C0", linewidths=0.5)

        plt.savefig(os.path.join(grids_path, f"{i}.png"), dpi=300)


        for k, v in {"I":I, "J":J}.items(): 

            fig = plt.figure(figsize=(8, 8)) 
            fig.add_subplot(1,1,1)
            plt.title(k)
            plt.imshow(v.squeeze(0).cpu().permute(1, 2, 0).data)
            plt.savefig(os.path.join(orig_path, f"{k}_{i:03d}.png"), dpi=300)

        fig = plt.figure(figsize=(8, 8)) 
        fig.add_subplot(1,1,1)
        plt.title("J_deformed")
        plt.imshow(Jw.squeeze(0).cpu().permute(1, 2, 0).data)
        plt.savefig(os.path.join(samples_path, f"Jw_{i:03d}.png"), dpi=300) 




if __name__ == "__main__":


    # dataset = FIREDataset("FIRE")
    dataset = Knee()
    # dataset = MNIST()


    # hparams_mnist = {
    #     "k": 2, 
    #     "down_factor": 1, 
    #     "lam": 2e1, 
    # }
    hparams_knee = {
        "k": 15, 
        "down_factor": 0.15, 
        "lam": 1e1, 
    }

    train(dataset, "OUTPUT_resnet", hparams_knee)