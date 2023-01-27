import torch
import numpy as np
import math

from numpy import matmul
from torch.nn.utils import prune

import NetworkHandler
print("whiteboxdetection")
directory = "whitebox_True_fine_tunedFalse_pruningFalse_batch_size32_lr0.01_momentum0.9_patience5"
def sigma(x):
    return 1/(1+math.exp(-x))

def DetectWBWatermark (Key,model):
    W=NetworkHandler.flattened_weight(model,"conv4.0.weight")
    S=NetworkHandler.projection(Key,W)
    return torch.round(S),S

def MatchAccuracy(tensor1,tensor2):
    goodcases = 0
    length=0
    for i,x in enumerate(tensor1):
        length +=1
        if tensor2[i].item() == x :
            goodcases+=1
    return goodcases*100/length
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")
net = NetworkHandler.ResNet().to(device)
if "pruningTrue" in directory :
    module = net.conv1[0]
    prune.l1_unstructured(module, name="weight", amount=0.3)
net.load_state_dict(torch.load(directory + "/model.pt"))
net_dimension = sum(p.numel() for p in net.parameters())

T = 64
watermark = torch.load(directory + "/watermark.pt")
X = torch.load(directory + "/X_key.pt")
Xp = NetworkHandler.X_creation(256*3*3, X_dimension=T, X_model="random").to(device)
print("Correct X :",X)

S_round,S=DetectWBWatermark(X,net)
print("S :",S)
print("DetectWBWatermark")
#print("Correct X :",X)
print("Result :",S_round)
print("Watermark :", watermark)
print("Match accuracy : ", MatchAccuracy(S_round,watermark), "%")

Sp_round,Sp=DetectWBWatermark(Xp,net)
print("Wrong WBWatermark")
print("Result :",Sp_round)
print("Watermark :", watermark)
print("Match accuracy : ", MatchAccuracy(Sp_round,watermark), "%")


def print_net(net):
    for name, parameters in net.named_parameters():
        print(name, ':', parameters.size())
    return
