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

from unet import SinusoidalPositionEmbeddings, UNet, PhiNet
from runet import RPhiNet
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


class FIREDataset(Dataset): 

    def __init__(self, fire_dir): 
        super().__init__() 
        self.dir = fire_dir 
        self.img1_paths = sorted(glob(os.path.join(fire_dir, "Images", "*_1.jpg")))
        self.img2_paths = sorted(glob(os.path.join(fire_dir, "Images", "*_2.jpg")))

        self.img1_ids = [x.split("/")[-1].split("_")[0] for x in self.img1_paths]
        self.img2_ids = [x.split("/")[-1].split("_")[0] for x in self.img2_paths]

        self.ground_truth_base_path = os.path.join(fire_dir, "Ground Truth")
        self.mask_path = os.path.join(fire_dir, "Masks", "mask.png")

        self.mask = io.imread(self.mask_path).astype(np.float32) / 255. 
        self.mask = torch.from_numpy(self.mask)[None, None, ...].to(DEVICE)

    def get_ground_truth_path(self, name:str) -> np.ndarray: 
        return os.path.join(self.ground_truth_base_path, f"control_points_{name}_1_2.txt")
    
    def get_gt_points(self, name:str) -> Tuple[np.ndarray, np.ndarray]: 
        gt_path = self.get_ground_truth_path(name) 

        with open(gt_path, "r") as f: 
            lines = f.readlines()

        data = np.array([l.split(" ") for l in lines]) 

        # first two columns is gt, the other two are starting points
        gt = data[:, :2]
        start = data[:, 2:]
        return gt, start

    def __getitem__(self, idx:int) -> Tuple[torch.Tensor, 
                                            torch.Tensor, 
                                            torch.Tensor, 
                                            Tuple[np.ndarray, np.ndarray]]: 
        img_id = self.img1_ids[idx]
        assert img_id == self.img2_ids[idx] 

        img1_path = self.img1_paths[idx]
        img2_path = self.img2_paths[idx] 

        I = io.imread(img1_path).astype(np.float32) / 255.0 
        J = io.imread(img2_path).astype(np.float32) / 255.0 

        h, w = 512, 512

        I_t = torch.from_numpy(I).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        J_t = torch.from_numpy(J).permute(2, 0, 1).unsqueeze(0).to(DEVICE)


        x = torch.linspace(-1, 1, w).to(DEVICE)
        y = torch.linspace(-1, 1, h).to(DEVICE)
        xy = torch.meshgrid(x, y, indexing="xy")
        xy = torch.stack(xy, dim=2).unsqueeze(0)

        I_t = F.grid_sample(I_t, xy, mode="bilinear", padding_mode="reflection", align_corners=False)
        J_t = F.grid_sample(J_t, xy, mode="bilinear", padding_mode="reflection", align_corners=False)


        gt, starting = self.get_gt_points(img_id) 

        return I_t, J_t, xy, (gt, starting)



    def __len__(self): 
        return len(self.img1_ids)


def get_dataloader() -> DataLoader:

    # The dataset should output I, J, xy_
    pass 



def train(dataset):

    learning_rate = 1e-3

    for i, item in enumerate(dataset):
        if i < 2: 
            continue

        network = RPhiNet(input_channels=6).to(DEVICE)
        optimizer = optim.Adam(network.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        optimizer1 = optim.Adam(network.parameters(), lr=0.01)
        scaler = torch.amp.GradScaler()
        torch.cuda.empty_cache()

        I, J, xy, points = item

        mask = dataset.mask


        pre_align = Homography() 
        trainer = Trainer(pre_align, 
                          lr=1e-3, 
                          levels=4, 
                          steps_per_epoch=100, 
                          loss_fn=dv_loss)

        trainer.register(I, J)
        pre_align = trainer.Hnet

        with torch.no_grad(): 
            J_pre, H = pre_align(J)


        # identity loss
        t0 = torch.zeros(1).to(DEVICE)
        for epoch in range(500):
            optimizer1.zero_grad()
            with torch.amp.autocast(DEVICE.type, dtype=torch.bfloat16):
                t = torch.rand(1).to(DEVICE)
                xyf = network(I,J_pre,xy,t0,t)
                xyr = network(I,J_pre,xy,t0,-t)
                loss = torch.mean((xyf-xy)**2)+torch.mean((xyr-xy)**2)
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
        lam=1e1
        nEpoch = 40
        batchsize = 1
        for epoch in range(nEpoch):

            res = np.sort((0.9)*np.random.rand(L-1)+0.1)
            res = np.concatenate((res,[1.0]))
            for level in range(L):
                optimizer.zero_grad()
                for step in range(batchsize):
                    loss,ss_loss = network.loss(I,J_pre,xy,res[level])
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
            loss,ss_loss = network.loss(I,J_pre,xy,res[level])
            total_loss = loss+lam*ss_loss
            print("Resolution:",L-1-level,"Epoch:",nEpoch,"Total loss:","{:.6f}".format(total_loss.item()),
                                "image loss:","{:.6f}".format(loss.item()),
                                "Constraint loss:","{:.10f}".format(ss_loss.item()),
            )


        # TEST 

        J_pre, H = pre_align(J) 
        xy_flattened = xy.reshape(-1, 2)
        xy_pre = (H @ torch.cat([xy_flattened, torch.ones_like(xy_flattened[:, -1:])], dim=1).T).T
        xy_pre = xy_pre[:, :2].reshape(1, xy.shape[1], xy.shape[2], 2)


        xyd = network(I,J_pre,xy,torch.zeros(1).to(DEVICE),1.0*torch.ones(1).to(DEVICE))

        xyd_r = network(I,J_pre,xy,torch.zeros(1).to(DEVICE),-1.0*torch.ones(1).to(DEVICE))

        Jw = F.grid_sample(J_pre,xyd,padding_mode='reflection',align_corners=True)
        Iw = F.grid_sample(I,xyd_r,padding_mode='reflection',align_corners=True)


        r1 = (J-I).max() - (J-I).min()
        r2 = (J_pre-I).max() - (J_pre-I).min()
        r3 = (Jw-I).max() - (Jw-I).min()

        m1 = (J-I).min()
        m2 = (J_pre-I).min()
        m3 =  (Jw-I).min()
        
        max_range = torch.max(torch.tensor([r1, r2, r3])).detach().cpu().item() 
        min_val = torch.min(torch.tensor([m1, m2, m3])).detach().cpu().item()

        imgI = I[0].permute(1, 2, 0).detach().cpu().numpy() 
        imgJ = J[0].permute(1, 2, 0).detach().cpu().numpy() 
        imgJ_pre = J_pre[0].permute(1, 2, 0).detach().cpu().numpy() 
        imgJw = Jw[0].permute(1, 2, 0).detach().cpu().numpy() 

        mi_I_J = histogram_mutual_information(imgI, imgJ)
        mi_I_J_pre = histogram_mutual_information(imgI, imgJ_pre)
        mi_I_Jw = histogram_mutual_information(imgI, imgJw)


        fig=plt.figure(figsize=(16,4))

        fig.add_subplot(1,3,1)
        plt.title(f"J-I before Reg. MI: {mi_I_J:.5f}")
        plt.imshow((((J-I) - min_val)/max_range).squeeze().cpu().permute(1, 2, 0).data)

        fig.add_subplot(1,3,2)
        plt.imshow((((J_pre-I) - min_val)/max_range).squeeze().cpu().permute(1, 2, 0).data)
        plt.title(f"J_pre-I after Reg. MI: {mi_I_J_pre:.5f}")


        fig.add_subplot(1,3,3)
        plt.imshow((((Jw-I) - min_val)/max_range).squeeze().cpu().permute(1, 2, 0).data)
        plt.title(f"Jw-I after Reg. MI: {mi_I_Jw:.5f}")


        plt.savefig(f"registration_results.png", dpi=300)

        fig=plt.figure(figsize=(5,5))
        d_ = (xyd - xy).squeeze()
        #fig=plt.figure(figsize=(5,5))
        fig.add_subplot(1,1,1)
        plt.quiver(d_.cpu().data[::5,::5,0], d_.cpu().data[::5,::5,1],color='r')
        plt.axis('equal')
        plt.title("Forward")

        # d__ = (xyd_r - xy_).squeeze()
        # #fig=plt.figure(figsize=(5,5))
        # fig.add_subplot(1,2,2)
        # plt.quiver(d__.cpu().data[::2,::2,0], d__.cpu().data[::2,::2,1],color='r')
        # plt.axis('equal')
        # plt.title("Reverse")
        # plt.show()

        plt.savefig("flow_vectors.png", dpi=300)


        break

if __name__ == "__main__":


    dataset = FIREDataset("FIRE")


    train(dataset)