'''
模型保存
'''
import torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=False)

# 保存方式1
torch.save(vgg16,"vgg16_method1.pth")

# 保存方式2,模型参数（推荐）
torch.save(vgg16.state_dict(), "vgg16_mothod2.pth")