import torch
import torchvision
from tensorboardX import SummaryWriter
from torch import nn
from torch.nn import Conv2d, Sequential, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from model1 import *

train_data = torchvision.datasets.CIFAR10("../data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

train_data_len = len(train_data)
test_data_len = len(test_data)

print("训练集长度：{}".format(train_data_len))
print("测试集长度:{}".format(test_data_len))

# Dataloader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, 1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, 1, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )
    def forward(self, x):
        x = self.model1(x)
        return x

tudui = Tudui()
if torch.cuda.is_available():
    tuidui = tudui.cuda()

# 损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 优化器
learn_rate = 0.01
# learn_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learn_rate)

# 设置训练网络的参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练轮次
epoch = 10

# 添加tensorboard
writer = SummaryWriter("../logs")

for i in range(epoch):
    print("-------第{}轮训练---------".format(i + 1))
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        output = tudui(imgs)
        loss = loss_fn(output, targets)

        # 优化器
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1

        if total_train_step % 100 == 0:
            print("训练次数:{},loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            output = tudui(imgs)
            loss = loss_fn(output, targets)
            total_test_loss = total_test_loss + loss.item()
    print("整体测试上的loss{}".format(total_test_loss))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(tudui.state_dict(), "tudui{}.pth".format(i))

writer.close()


