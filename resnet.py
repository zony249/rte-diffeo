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

from homography import dv_loss

random_seed = 1
torch.backends.cudnn.enabled = True
torch.manual_seed(random_seed)

# Checking GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



class Block(nn.Module): 
    def __init__(self, step_size, hidden=8): 
        super().__init__() 
        self.conv1a = nn.Conv2d(in_channels=2, out_channels=hidden, kernel_size=8, padding="same")
        self.conv1b = nn.Conv2d(in_channels=hidden, out_channels=2, kernel_size=1, padding="same")
        
        # self.conv2a = nn.Conv2d(in_channels=2, out_channels=hidden, kernel_size=1, padding="same")
        # self.conv2b = nn.Conv2d(in_channels=hidden, out_channels=2, kernel_size=1, padding="same")
        self.step_size = step_size
    def forward(self, x):    
        ha = F.elu(self.conv1a(x)) 
        va = self.conv1b(ha)

        # hb = F.elu(self.conv2a(x)) 
        # vb = self.conv2b(hb)

        v = va #+ vb

        return self.step_size * v + x, v 
    
class ResNet(nn.Module): 
    def __init__(self, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([Block(1/num_layers) for _ in range(num_layers)]) 
        self.num_layers = num_layers
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1), 
            nn.SiLU(), 
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1)
        )
    def encode(self, I, J): 
        Ie = self.encoder(I)
        Je = self.encoder(J)
        return Ie, Je
    
    def forward(self, J, xy): 
        vs = []
        xys = []
        for l in self.layers: 
            xy, v = l(xy.permute(0, 3, 1, 2))
            xy = xy.permute(0, 2, 3, 1)
            xys.append(xy)
            vs.append(v)
        vs = torch.stack(vs, dim=0)
        J = F.grid_sample(J, xy, mode="bilinear", padding_mode="reflection", align_corners=False)
        return J, xy, {"vs": vs, "xys": xys}
    def loss(self, I, J, xy): 
        Jw, xyL, past = self(J, xy) 
        image_loss = dv_loss(Jw, I, encode_function=self.encode) 

        vs = past["vs"]
        xys = past["xys"]

        l, n, c, h, w = vs.shape 
        vs = vs.reshape(l, c, h, w) 
        laplacian = torch.tensor([[0, -1, 0], 
                                  [-1, 4, -1], 
                                  [0, -1, 0]], dtype=torch.float32, device=device)[None, None, ...]
        laplacian = torch.cat([laplacian, laplacian], dim=0)
        Lvs = F.conv2d(vs, laplacian, stride=1, padding=1, groups=2)
        


        flow_loss = (Lvs.norm(dim=1)**2).sum(dim=0).mean()/self.num_layers # [L, C, H, W]

        for intermed in xys: 
            image_loss += dv_loss(F.grid_sample(J, intermed, "bilinear", "reflection", False), I, encode_function=self.encode) / self.num_layers

        return image_loss, flow_loss
