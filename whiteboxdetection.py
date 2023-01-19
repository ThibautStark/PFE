import torch
import numpy as np
import math

from numpy import matmul

import NetworkHandler
print("whiteboxdetection")

def sigma(x):
    return 1/(1+math.exp(-x))

def DetectWBWatermark (Key,model):
    W=NetworkHandler.flattened_weight(model,"conv4.0.weight")
    S=NetworkHandler.projection(Key,W)
    #print(S)
    S=torch.tensor([sigma(S[i]) for i in range(S.size()[0])])
    return S

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")
net = NetworkHandler.ResNet().to(device)
net.load_state_dict(torch.load("CIFAR10_images/whitebox_Truebatch_size32_lr0.01_momentum0.9_patience5model.pt"))
net_dimension = sum(p.numel() for p in net.parameters())

T = 64
watermark = torch.tensor(np.random.choice([0, 1], size=(T), p=[1. / 3, 2. / 3]), device=device)
X = torch.load("X_key.pt")
Xp = NetworkHandler.X_creation(256*3*3, X_dimension=T, X_model="random").to(device)
S=DetectWBWatermark(X,net)
print("DetectWBWatermark")
#print(X)
print(S)
Sp=DetectWBWatermark(Xp,net)
print("Wrong WBWatermark")
#print(Xp)
print(Sp)
def print_net(net):
    for name, parameters in net.named_parameters():
        print(name, ':', parameters.size())
    return
