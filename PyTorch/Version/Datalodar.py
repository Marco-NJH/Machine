import torchvision
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

test_data = torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=True)
#测试数据集中第一张图片及target
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("p10")
step = 0
for data in test_loader:
    imgs,target = data
    # print(img.shape)
    # print(target)
    writer.add_images("test_data_true",imgs,step)
    step = step+1

writer.close()