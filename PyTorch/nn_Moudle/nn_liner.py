import torch
import torch.nn as nn
import torchvision
from torch.nn import Conv2d, MaxPool2d, Linear
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../data",train=False,transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataLoader = DataLoader(dataset,batch_size=64)

class Tudui(nn.Module):

    def __init__(self):
        super(Tudui,self).__init__()
        self.liner1 = Linear(196608,10)

    def forward(self, input):
        output = self.liner1(input)
        return output


tudui = Tudui()
for data in dataLoader:
    imgs, target = data
    print(imgs.shape)
    # output = torch.reshape(imgs,(1,1,1,-1))
    output = torch.flatten(imgs)
    print(output.shape)
    output =  tudui(output)
    print(output.shape)