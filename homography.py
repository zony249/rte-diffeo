import os 
import numpy as np 
from typing import Tuple, List, Optional, Union, Literal
from PIL import Image
from copy import deepcopy

import torch 
from torch import nn 
import torch.nn.functional as F
from torch.optim import AdamW, LBFGS
import cv2


import matplotlib.pyplot as plt 

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Tnet(nn.Module): 
    def __init__(self, 
                 in_channels, 
                 features=16): 
        super().__init__() 
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=1, padding="same", bias=False)
        self.conv2 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=1, padding="same", bias=False)
        self.in_channels = in_channels
    def forward(self, I, J): 
        """
        I, J: torch.Tensor[N, C, H, W]
        """
        
        x = self.conv1(I)
        x = F.relu(x) + x 
        x = self.conv2(x) 

        y = self.conv1(J)
        y = F.relu(y) + y 
        y = self.conv2(y) 

        return x, y
    def perturb_weights(self, spread=0.02): 
        with torch.no_grad(): 
            self.conv1.weight = nn.Parameter(self.conv1.weight + torch.randn_like(self.conv1.weight) * spread)
            self.conv2.weight = nn.Parameter(self.conv2.weight + torch.randn_like(self.conv2.weight) * spread)
    def shrink_weights(self, lmbda=0.5): 
        with torch.no_grad(): 
            self.conv1.weight = nn.Parameter(self.conv1.weight * lmbda)
            self.conv2.weight = nn.Parameter(self.conv2.weight * lmbda)

class Homography(nn.Module): 
    def __init__(self, input_channels_per_image=3, features=32): 
        super().__init__()  
        self.basis = torch.zeros((8, 3, 3), device=DEVICE)
        self.basis[0,0,2] = 1. 
        self.basis[1,1,2] = 1. 
        self.basis[2,0,1] = 1. 
        self.basis[3,1,0] = 1.
        self.basis[4,0,0], self.basis[4,1,1] = 1., -1. 
        self.basis[5,1,1], self.basis[5,2,2] = -1., 1.
        self.basis[6,2,0] = 1. 
        self.basis[7,2,1] = 1. 

        self.v = nn.Parameter(torch.zeros((8,1,1), device=DEVICE), requires_grad=True) 

        self.features = features
        self.encode = Tnet(in_channels=input_channels_per_image, features=features).to(DEVICE)


    def forward(self, I: torch.Tensor, size=None) -> Tuple[torch.Tensor, torch.Tensor]: 
        """
        I: torch.Tensor[B, C, H, W]
        """
        H = self.Mexp(self.basis, self.v) 

        if size is None:  
            h, w = I.shape[-2], I.shape[-1]
        else: 
            h, w = size
        x = torch.linspace(-1, 1, w, device=DEVICE)
        y = torch.linspace(-1, 1, h, device=DEVICE) 
        xx, yy = torch.meshgrid(x, y, indexing='xy') 
        grid = torch.stack([xx, yy], dim=-1).reshape(-1, 2).T
        grid_t = H @ torch.cat([grid, torch.ones_like(grid[-1:, :], device=DEVICE)], dim=0)
        grid_t /= grid_t[2:, :].clone()
        xx_t, yy_t = grid_t[0, :].reshape(xx.shape), grid_t[1, :].reshape(yy.shape)

        grid_sample = torch.stack([xx_t, yy_t], dim=-1)[None, :, :, :]
        J = F.grid_sample(I, grid_sample, align_corners=False)
        return J, H 
    
    def Mexp(self, B, v): 
        A = torch.eye(3, device=DEVICE)
        n_fact = 1
        H = torch.eye(3, device=DEVICE)
        for i in range(20): 
            A = (v * B).sum(dim=0) @ A
            n_fact = max(i, 1) * n_fact
            A /= n_fact
            H += A 
        return H / H[2, 2]

    def log_factorial(self, x):
        return torch.lgamma(x + 1)
    def factorial(self, x): 
        return torch.exp(self.log_factorial(x))

    def encode_image(self, I:torch.Tensor, J:torch.Tensor) -> torch.Tensor: 

        h, w = I.shape[2:]  
        
        b = I.shape[0] 
        c = I.shape[1]

        return self.encode(I, J)

    def reset_encoder(self): 
        for param in self.encode.parameters():
            nn.init.xavier_normal_(param)

    def shrink_perturb(self, lmbda=0.5, spread=0.02): 
        self.encode.shrink_weights(lmbda)
        self.encode.perturb_weights(spread)



class Trainer: 
    def __init__(self, 
                 Hnet, 
                 lr=1e-3,
                 levels=3,
                 steps_per_epoch=10, 
                 encode_weight_decay=3e-4, 
                 shrink_lmbda=0.99, 
                 perturb_spread=1e-5, 
                 loss_fn=None): 
        self.Hnet = Hnet
        self.optim = torch.optim.AdamW([{"params": self.Hnet.v, "lr": lr}, 
                            {"params": self.Hnet.encode.parameters(), "lr": lr*10, "weight_decay": encode_weight_decay}])
        self.levels = levels 
        self.steps_per_epoch = steps_per_epoch
        self.loss_fn = loss_fn
        if self.loss_fn is None: 
            self.loss_fn = mse

        self.lr = lr
        self.encode_weight_decay = encode_weight_decay
        self.shrink_lmbda = shrink_lmbda
        self.perturb_spread = perturb_spread

    def convert_img_to_torch(self, imgI: np.ndarray) -> torch.Tensor: 

        if imgI.ndim == 2: 
            I = torch.from_numpy(imgI).float().to(DEVICE)[None, None, :, :]
        elif imgI.ndim == 3:  
            I = torch.permute(torch.from_numpy(imgI), 
                            (2, 0, 1)).float().to(DEVICE)[None, :, :, :]
        return I 

    def register(self, I: torch.Tensor, J: torch.Tensor): 
        # self.Hnet.reset_encoder()
        self.optim = torch.optim.AdamW([{"params": self.Hnet.v, "lr": self.lr}, 
                            {"params": self.Hnet.encode.parameters(), "lr": self.lr*10, "weight_decay": 3e-4}])

        with torch.no_grad(): 
            J_w, H = self.Hnet(J)
            pre_reg_J = J_w.detach().cpu().numpy()[0, 0]
            # plt.imshow(pre_reg_J)
            # plt.savefig("pre-registration-J.png")

        scales = 2.0**torch.arange(-self.levels, 1, device=DEVICE)


        for level in range(1, self.levels + 1): 
            self.optim.zero_grad()
            I_s, J_s = self.scale(I, scales[level]), self.scale(J, scales[level])
            scale_J = J_s.detach().cpu().numpy()[0, 0]
            # plt.imshow(scale_J)
            # plt.savefig(f"scale_{scales[level]}--{level}.png")

            h_, w_ = I_s.shape[2:]

            for step in range(self.steps_per_epoch): 
                self.Hnet.zero_grad()                
                J_w, H = self.Hnet(J_s, size=(h_, w_))
                loss = self.loss_fn(I_s, J_w, encode_function=self.Hnet.encode_image)
                loss.backward() 

                cur_v = deepcopy(self.Hnet.v)

                self.optim.step() 

                new_v = deepcopy(self.Hnet.v)

                # if (new_v - cur_v).norm() / H.norm() < 1e-3:
                #     break
                print(loss.item())
        return J_w, H

    def scale(self, I, s): 


        k_x = torch.linspace(-1/s, 1/s, int(2/s - 1), device=DEVICE) 
        xx, yy = torch.meshgrid(k_x, k_x, indexing="xy")
        kernel = torch.exp(-0.5 * (xx**2 + yy**2)/s**2)
        kernel /= kernel.sum()
        kernel = kernel[None, None, :, :]
        kernel = torch.zeros((I.shape[1], 1, 1, 1), device=DEVICE) + kernel
        I_smooth = F.conv2d(I, kernel, groups=I.shape[1], padding="same") 


        h, w = I.shape[-2], I.shape[-1]

        x = torch.linspace(-1, 1, int(s*w), device=DEVICE)
        y = torch.linspace(-1, 1, int(s*h), device=DEVICE) 

        xx, yy = torch.meshgrid(x, y, indexing='xy') 
        grid_s = torch.stack([xx, yy], dim=-1)[None, :, :, :]

        Is = F.grid_sample(I_smooth, grid_s, align_corners=False)

        return Is
    
    def solve_initial(self, H_ref, template): 
        """
        points: torch.Tensor[4, 2]
        assumption: points are not origin-centered
        """

        params = [{"params": self.Hnet.v}, {"params": self.Hnet.encode.parameters(), "lr": 1e-2,  "weight_decay": 1e-2}]
        initial_optim = torch.optim.SGD(params, lr=1e-1) 

        for i in range(1000): 
            initial_optim.zero_grad() 
            H = self.Hnet.Mexp(self.Hnet.basis, self.Hnet.v) 
            loss = ((H - H_ref)**2).sum() + self.loss_fn(template, template, encode_function=self.Hnet.encode_image)

            loss.backward() 
            initial_optim.step() 

            print(f"loss: {loss.item():.4f}")

        print("Reference H", H_ref)
        print("Learned H", H)

        initial_optim = None
        return self.Hnet


def mse(targets: torch.Tensor, inputs: torch.Tensor, encode_function=None): 
    mse_loss = F.mse_loss(inputs.flatten(), targets.flatten()) 
    return mse_loss 

def contrastive_sim(targets:torch.Tensor, inputs: torch.Tensor,): 


    uv = targets @ inputs.transpose(-2, -1) 

    uu = targets.norm(dim=-1, keepdim=True)
    vv = inputs.transpose(-2, -1).norm(dim=-2, keepdim=True)

    cosines = uv / (uu * vv + 1e-8) 

    pos = torch.diagonal(cosines, dim1=-2, dim2=-1).sum() / cosines.shape[-1]
    neg = torch.triu(cosines, diagonal=1).sum() / (cosines.shape[-1] * (cosines.shape[-2]-1)/2)

    return -pos + neg


def dv_loss(I, J, encode_function=None): 

    b, c, h, w = J.shape

    if encode_function is not None:
        Ie, Je = encode_function(I, J)
        first_term = ((Ie - Je)**2).mean(dim=1).mean()
    else: 
        T = ((I - J)**2).sum(dim=1) 
        first_term = T.mean()

    rand_idx = torch.randperm(h*w) 
    J_rand = J.reshape(b, c, h*w)[:, :, rand_idx].reshape(b, c, h, w)

    if encode_function is not None:
        Ie, Je_rand = encode_function(I, J_rand)
        second_term = torch.exp(-((Ie - Je_rand)**2).mean(dim=1))
        second_term = torch.log(second_term.mean())
    else:
        T_rand = -((I - J_rand)**2)
        second_term = torch.exp(T_rand)
        second_term = torch.log(second_term.mean())
    # print("first term", first_term)
    # print("second term", second_term)
    return first_term + second_term 






def histogram_mutual_information(image1, image2):
    hgram, x_edges, y_edges = np.histogram2d(image1.ravel(), image2.ravel(), bins=100)
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))





if __name__ == "__main__": 
    
    Hnet = Homography(input_channels_per_image=1, features=32).to(DEVICE) 
    
    # I = torch.zeros((1, 1, 5, 6), device=DEVICE)

    imgI = Image.open("knee1.bmp")
    imgI = np.array(imgI) / 255.
    imgJ = Image.open("knee2.bmp")
    imgJ = np.array(imgJ) / 255.
    
    plt.imshow(imgI)
    plt.savefig("imgI.png")

    plt.imshow(imgJ)
    plt.savefig("imgJ.png")

    # Hnet = Homography() 




    trainer = Trainer(
        Hnet, 
        lr=1e-2, 
        levels=4, 
        steps_per_epoch=100, 
        loss_fn=mse, 
    )
    J, H = trainer.register(imgI, imgJ) 

    registered_J = J.detach().cpu().numpy()[0, 0]
    plt.imshow(registered_J)
    plt.savefig("registered_J.png")


    pre_mi = histogram_mutual_information(image1=imgI, image2=imgJ)
    post_mi = histogram_mutual_information(image1=imgI, image2=registered_J)

    print(f"MI before registering: {pre_mi}")
    print(f"MI after registering: {post_mi}")


    points = torch.tensor([[100, 100], 
              [104, 300], 
              [200, 330], 
              [200, 110]], device=DEVICE)
              
    # trainer.solve_initial(points)