import torchvision
from tensorboardX import SummaryWriter

dataset_transfoms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
trans_set = torchvision.datasets.CIFAR10(root="./dataset",train=True,transform=dataset_transfoms,download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=dataset_transfoms,download=True)
# print(test_set[0])
# print(test_set.classes)
#
# img,target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()
write = SummaryWriter("p10")
for i in range(10):
    img,targe = test_set[i]
    write.add_image("test_set",img,i)

write.close()
