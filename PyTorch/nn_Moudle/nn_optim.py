import torch.optim
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../data",train=False,transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataLoader = DataLoader(dataset,batch_size=1)
class Tudui(nn.Module):
    def __init__(self) :
        super(Tudui,self).__init__()


        self.model1 = Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self,x):

        x = self.model1(x)
        return x

loss = nn.CrossEntropyLoss()
tudui = Tudui()
optim = torch.optim.SGD(tudui.parameters(),lr=0.01)
for i in range(20):
    run_loss = 0.0
    for data in dataLoader:
        imgs, target = data
        outputs = tudui(imgs)
        result = loss(outputs,target)
        optim.zero_grad()
        result.backward()
        optim.step()
        run_loss = run_loss+result
    print(run_loss)
