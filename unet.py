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

random_seed = 1
torch.backends.cudnn.enabled = True
torch.manual_seed(random_seed)

# Checking GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



class SinusoidalPositionEmbeddings(nn.Module):
  def __init__(self, dim):
    super().__init__()

    self.dim = dim

  def forward(self, time):
    device = time.device
    half_dim = self.dim // 2
    embeddings = math.log(10000) / (half_dim - 1)
    embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
    embeddings = time[:, None] * embeddings[None, :]
    embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

    return embeddings




class Block2d(nn.Module):
  def __init__(self, in_channels, out_channels, time_emb_dim, up=False):
    super().__init__()

    self.time_mlp =  nn.Linear(time_emb_dim, out_channels)

    if up:
      # up-sampling (decoder part)
      self.conv1 = nn.Conv2d(2*in_channels, out_channels, 3, padding=1)
      self.transform = nn.ConvTranspose2d(out_channels, out_channels, 4, 2, 1)
      #self.transform = nn.Sequential(nn.Conv2d(out_channels,out_channels,3,padding=1),
      #                               torch.nn.Upsample(scale_factor=2, mode='bilinear'))
    else:
      # down-sampling (encoder part)
      self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
      self.transform = nn.Conv2d(out_channels, out_channels, 4, 2, 1)

    self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
    self.bnorm1 = nn.BatchNorm2d(out_channels)
    self.bnorm2 = nn.BatchNorm2d(out_channels)

    self.t_embed = SinusoidalPositionEmbeddings(time_emb_dim)

  def forward(self, x:torch.Tensor, t: torch.Tensor):
    # First Conv
    h = self.bnorm1(F.silu(self.conv1(x)))
    # Time embedding

    # print("t", t.shape)
    # print("t", t.shape)
    # t_enc = self.t_embed(t)
 

    # print("t_enc", t_enc.shape)
    # print("x", x.shape)




    time_emb = F.silu(self.time_mlp(t[1][None, :] - t[0][None, :]))
    # time_emb = F.silu(self.time_mlp(t[1][None, :]) - self.time_mlp(t[0][None, :]))
    # Extend last 2 dimensions
    time_emb = time_emb[(..., ) + (None, ) * 2]
    # Add time channel
    h = h + time_emb
    # Second Conv
    h = self.bnorm2(F.silu(self.conv2(h)))

    # Down or Upsample
    out = self.transform(h)

    return out




class UNet(nn.Module):
  """
  A simplified variant of the Unet architecture.
  """
  def __init__(self, in_channels=1, out_channels=1, time_emb_dim=64):
    super().__init__()

    down_channels = (8, 16, 64, 512)
    up_channels = (512, 64, 16, 8)


    # Time embedding
    self.time_mlp = nn.Sequential(
        SinusoidalPositionEmbeddings(time_emb_dim),
        nn.Linear(time_emb_dim, time_emb_dim),
        nn.SiLU()
    )

    # Initial projection
    self.conv0 = nn.Conv2d(in_channels, down_channels[0], 3, padding=1)

    # Downsample
    self.downs = nn.ModuleList([Block2d(down_channels[i], down_channels[i+1], time_emb_dim) for i in range(len(down_channels)-1)])

    # Upsample
    self.ups = nn.ModuleList([Block2d(up_channels[i], up_channels[i+1], time_emb_dim, up=True) for i in range(len(up_channels)-1)])

    # Final projection
    self.output = nn.Conv2d(up_channels[-1], out_channels, 1)

  def forward(self, x, timestep):
    # Embedd time
    t = self.time_mlp(timestep)

    # Initial conv
    x = self.conv0(x)
    # Unet
    residual_inputs = []
    for down in self.downs:
        x = down(x, t)
        residual_inputs.append(x)
    for up in self.ups:
        residual_x = residual_inputs.pop()

        # Add residual x as additional channels
        x = torch.cat((x, residual_x), dim=1)
        x = up(x, t)
    return self.output(x)



class PhiNet(nn.Module):
  def __init__(self, input_channels=2):
    super(PhiNet, self).__init__()
    self.net = UNet(input_channels,2).to(device)

  def forward(self,I,J,xy0,t0,t):
    uin = torch.cat((I,J),dim=1)
    phi = xy0 + (t-t0)*self.net(uin,torch.cat((t0,t))).permute(0,2,3,1)
    # print("xy0", xy0.shape)
    # print("uin", uin.shape)
    # print("torch.cat((t0, t))", torch.cat((t0,t)).shape)
    # print("")
    return phi

  def velocity(self,I,J,xy):
    uin = torch.cat((I,J),dim=1)
    return self.net(uin,torch.zeros(1).to(device)).permute(0,2,3,1)

  # forward-backward transform and warp
  # Implements $J(\phi(t,x,y)) = I(\phi^{-1}(1-t,x,y)) = I(\phi(t-1,x,y))$ for $0 \le t \le 1$.
  # limited step scaling and squaring integration

  def loss(self,I,J,xy,res, mask=None):

    h_ = int(round(I.shape[2]*res))
    w_ = int(round(I.shape[3]*res))

    t0 = torch.zeros(1).to(device)
    t = torch.rand(1).to(device)

    phiJ = self(I, J, xy, t0, t)
    Jw = F.grid_sample(J,phiJ,padding_mode='reflection',align_corners=True)

    phiI = self(I, J, xy, t0, t-1)
    Iw = F.grid_sample(I,phiI,padding_mode='reflection',align_corners=True)

    if mask is not None: 
      maskw = F.grid_sample(mask.float(), phiI, padding_mode='zeros', align_corners=True)
      maskw_level = interpolate(maskw, size=(h_, w_), mode='bilinear', antialias=True) > 0.5 

    Jw_level = interpolate(Jw,size=(h_,w_),mode='bilinear',antialias=True)
    Iw_level = interpolate(Iw,size=(h_,w_),mode='bilinear',antialias=True)
    if mask is not None: 
      image_loss = torch.where(maskw_level, (Iw_level - Jw_level)**2, 0).sum() / (maskw_level.sum() * Iw.shape[1])
    else:
      image_loss = F.mse_loss(Jw_level,Iw_level) 

    phiI_ = self(I, J, xy, t, t-1) 
    phiI_ = F.grid_sample(phiI_.permute(0, 3, 1, 2), phiJ, padding_mode="reflection", align_corners=True).permute(0, 2, 3, 1)

    
    phiJ_ = self(I, J, xy, t-1, t)
    phiJ_ = F.grid_sample(phiJ_.permute(0, 3, 1, 2), phiI, padding_mode="reflection", align_corners=True).permute(0, 2, 3, 1)

    # print("phiJ", phiJ.shape)
    # print("phiJ_", phiJ_.shape)

    flow_loss = 0.5*(torch.mean((phiJ-phiJ_)**2)+torch.mean((phiI-phiI_)**2))

    return image_loss,flow_loss



