import torch
import torchvision
from PIL import Image

image_path = "../Picture/img.png"
image = Image.open(image_path)
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
# 创建网络模型
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear, Softmax


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
            # Softmax(),
            Linear(1024, 64),
            Linear(64, 10)
        )
    def forward(self, x):
        x = self.model1(x)
        return x

model = torch.load("tudui9.pth", map_location=torch.device('cpu'))
# model.load_state_dict(torch.load("tudui9.pth"))
print(model)
image = torch.reshape(image,(1,3,32,32))
# model.eval()
with torch.no_grad():
    output = model(image)
print(output)

print(output.argmax(1))