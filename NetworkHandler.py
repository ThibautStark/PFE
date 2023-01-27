import torch
from torchvision.models import ResNet
import torch.nn as nn
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.res1 = nn.Sequential(nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        ), nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.res2 = nn.Sequential(nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        ), nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        )

        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x) + x
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res2(x) + x
        x = self.classifier(x)

        return x

def X_creation(net_dimension,X_dimension=1,X_model="random"):
  if X_model=="random" :
    X=torch.rand((X_dimension,net_dimension))
  elif X_model=="diff":
    X=torch.zeros((X_dimension,net_dimension))
    for i in range (X.size()):
      X[i][0]=1
      X[i][1]=-1
  elif X_model == "direct":
    X=torch.zeros([X_dimension,net_dimension])
    for i in range (X.size()):
      X[i][0]=1
  return X


def flattened_weight(net, weights_name):
  '''
  :param net: aimed network
  :param weights_name: aimed layer's name
  :return: a vector of dimension CxKxK (flattened weight)
  '''

  for name, parameters in net.named_parameters():
      if weights_name in name:
          f_weights = torch.mean(parameters, dim=0)
          f_weights = f_weights.view(-1, )
  return f_weights

def projection(X, w):
    '''
    :param X: secret key matrix
    :param w: flattened weight
    :return: sigmoid of the matrix multiplication of the 2 inputs
    '''
    sigmoid_func = nn.Sigmoid()
    res = torch.matmul(X, w)
    sigmoid = sigmoid_func(res)
    return sigmoid

def loss_W(net, weights_name, X, watermark):
    '''
    :param net: aimed network
    :param weights_name: aimed layer's name
    :param X: secret key matrix
    :param watermark: the watermark
    :return: Uchida's loss
    '''
    loss = 0
    W = flattened_weight(net, weights_name)
    yj = projection(X, W)
    for i in range(len(watermark)):
        loss += watermark[i] * torch.log2(yj[i]) + (1 - watermark[i]) * torch.log2(1 - yj[i])
    return -loss/len(watermark)
