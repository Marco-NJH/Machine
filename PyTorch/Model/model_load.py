import torch


# 对应方式1加载模型
# model = torch.load("vgg16_method1.pth")
# print(model)

# 方式2,加载
import torchvision.models

vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_mothod2.pth"))
# model1 = torch.load("vgg16_mothod2.pth")
print(vgg16)