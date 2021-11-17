import torch
import torch.nn as nn
import torchvision
from torch.nn import Conv2d, MaxPool2d
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.nn import ReLU
from torch.nn import Sigmoid

# 非线性激活  Relu   Sigmoid
input = torch.tensor([[1,-0.5],
                      [-1,3]])
input = torch.reshape(input,(-1,1,2,2))
print(input.shape)

dataset = torchvision.datasets.CIFAR10("../data",train=False,transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataLoader = DataLoader(dataset,batch_size=64)
class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1 = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, input):
        # output = self.relu1(input)
        output = self.sigmoid(input)
        return output

tudui = Tudui()
# output = tudui(input)
# print(output)

writer = SummaryWriter("../logs")
step = 0
for data in dataLoader:
    imgs, target = data
    output = tudui(imgs)
    # print(imgs.shape)
    # print(output.shape)
    writer.add_images("input-xian",imgs,step)
    # output = torch.reshape(output,(-1,3,10,10))
    writer.add_images("output-xian",output,step)
    step = step+1

writer.close()
