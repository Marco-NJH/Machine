import torch
import torch.nn as nn
import torchvision
from torch.nn import Conv2d, MaxPool2d
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
# 池化
# input = torch.tensor([[1,2,0,3,1],
#                       [0,1,2,3,1],
#                       [1,2,1,0,0],
#                       [5,2,3,1,1],
#                       [2,1,0,1,1]],dtype=torch.float32)
# input = torch.reshape(input,(-1,1,5,5))
# print(input.shape)
dataset = torchvision.datasets.CIFAR10("../data",train=False,transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataLoader = DataLoader(dataset,batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = MaxPool2d(kernel_size=3,ceil_mode=False)

    def forward(self, input):
        output = self.maxpool(input)
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
    writer.add_images("input-chi",imgs,step)
    # output = torch.reshape(output,(-1,3,10,10))
    writer.add_images("output-chi",output,step)
    step = step+1

writer.close()