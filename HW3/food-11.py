# Import necessary packages.
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# 在进行半监督学习时，“ConcatDataset”和“子集”可能是有用的。
from tensorboardX import SummaryWriter
from torch.utils.data import ConcatDataset, DataLoader, Subset, dataloader
from torchvision.datasets import DatasetFolder

# 进度条
from tqdm.auto import tqdm

# It is important to do data augmentation in training.
# However, not every augmentation is useful.
# Please think about what kind of augmentation is helpful for food recognition.
train_tfm = transforms.Compose([
    # 将图像调整为固定形状(高度=宽度= 128)
    transforms.Resize((128, 128)),
    # You may add some transforms here.
    # ToTensor() 应该是最后一个转换。
    transforms.ToTensor(),
])

# 我们不需要增加测试和验证。
# 我们需要做的就是调整PIL图像的大小，并将其转换为张量。
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# 用于培训、验证和测试的批大小。
# 更大的批大小通常提供更稳定的梯度。
# 批处理较大的批处理。但是GPU内存有限，请仔细调整。
batch_size = 128

# 构建数据集。
# 参数“loader”告诉了torchvision如何读取数据
train_set = DatasetFolder("food-11/training/labeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
valid_set = DatasetFolder("food-11/validation", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
unlabeled_set = DatasetFolder("food-11/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
test_set = DatasetFolder("food-11/testing", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)

# 构建数据加载器。
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # 常用模块的参数:
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        # 输入图片大小 [3, 128, 128]
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]

        # 通过卷积层提取特征。
        x = self.cnn_layers(x)

        # 提取的特征映射必须在进行全连接层之前进行平坦处理。
        x = x.flatten(1)

        # 通过全连通层对特征进行变换，得到最终的对数。
        x = self.fc_layers(x)
        return x

def get_pseudo_labels(dataset, model, threshold=0.65):
    # 这个函数使用给定的模型生成数据集的伪标签。
    # 它返回一个DatasetFolder实例，其中包含预测置信度超过给定阈值的图像。
    # 不允许使用任何经过外部数据训练的模型进行伪标记。
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 确保模型处于eval模式。
    model.eval()
    # 定义softmax函数。
    softmax = nn.Softmax(dim=-1)

    # 按批次遍历数据集。
    for batch in tqdm(dataloader):
        img, _ = batch

        # 提出了数据
        # 使用 torch.no_grad() 加速前进过程。
        with torch.no_grad():
            logits = model(img.to(device))

        # 通过对logits应用softmax获得概率分布。
        probs = softmax(logits)

        # ---------- TODO ----------
        # 筛选数据并构建新的数据集。

    # # Turn off the eval mode.
    model.train()
    return dataset

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

# 初始化一个模型，并将其放在指定的设备上。
model = Classifier().to(device)
model.device = device

# 对于分类任务，我们使用交叉熵作为性能的度量。
criterion = nn.CrossEntropyLoss()

# 初始化优化器，您可以自己微调一些超参数，如学习速率。
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# 训练期的数量。
n_epochs = 80

# 是否进行半监督学习。
do_semi = False

# 添加tensorboard
writer = SummaryWriter("/logs")

for epoch in range(n_epochs):
    # ---------- TODO ----------
    # 在每个时期，为半监督学习重新标记未标记的数据集。
    # 然后，您可以结合标记数据集和伪标记数据集进行培训。
    if do_semi:
        # 对未标注数据使用训练模型获得伪标签。
        pseudo_set = get_pseudo_labels(unlabeled_set, model)

        # 构建一个新的数据集和一个用于培训的数据加载器。
        # 这只用于半监督学习。
        concat_dataset = ConcatDataset([train_set, pseudo_set])
        train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    # ---------- Training ----------
    # 在训练前确保模型处于训练模式。
    model.train()

    # 记录训练信息的。
    train_loss = []
    train_accs = []

    # 按批次迭代训练集。
    for batch in tqdm(train_loader):

        # 批处理由图像数据和相应的标签组成。
        imgs, labels = batch

        # 提出了数据。(确保数据和模型在同一台设备上。)
        logits = model(imgs.to(device))

        # 计算交叉熵损失。
        # 我们不需要在计算交叉熵之前应用softmax，因为它是自动完成的。
        loss = criterion(logits, labels.to(device))

        # 应该首先清除上一步中存储在参数中的渐变。
        optimizer.zero_grad()

        # 计算参数的梯度。
        loss.backward()

        # 修剪梯度规范的稳定训练。
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # 使用计算的梯度更新参数。
        optimizer.step()

        # 计算当前批处理的精度。
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # 记录损失和准确性。
        train_loss.append(loss.item())
        train_accs.append(acc)

    # 训练集的平均损耗和精度是记录值的平均值。
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)
    writer.add_scalar("train_loss", train_loss.item(), epoch)
    writer.add_scalar("train_acc", train_acc.item(), epoch)

    # 打印信息。
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    # ---------- Validation ----------
    # 确保该模型处于eval模式，以便某些模块(如dropout)被禁用并正常工作。
    model.eval()

    # 它们用于记录验证中的信息。
    valid_loss = []
    valid_accs = []

    # 按批次迭代验证集。
    for batch in tqdm(valid_loader):

        # 批处理由图像数据和相应的标签组成。
        imgs, labels = batch

        # 我们不需要在验证中使用梯度。
        # 使用torch.no_grad()可以加速向前进程
        with torch.no_grad():
          logits = model(imgs.to(device))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # 计算当前批处理的精度。
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # 记录损失和准确性。
        valid_loss.append(loss.item())
        valid_accs.append(acc)

    # 整个验证集的平均损失和准确性是记录值的平均值。
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)
    writer.add_scalar("valid_loss", valid_loss.item(), epoch)
    writer.add_scalar("valid_acc", valid_acc.item(), epoch)

    # Print the information.
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

    torch.save(model.state_dict(),"food{}.pth".format(epoch))

# 确保模型处于eval模式。
# 有些模块如Dropout或BatchNorm会影响模型是否处于训练模式。
model.eval()

# 初始化一个列表来存储预测。
predictions = []

# 按批次迭代测试集
for batch in tqdm(test_loader):
    # 批处理由图像数据和相应的标签组成。
    # 但在这里，变量“标签”是无用的，因为我们没有基本真理。
    # 如果打印出标签，你会发现它总是0。
    # 这是因为包装器(DatasetFolder)
    # 为每个批返回图像和标签，
    # 所以我们必须制作假标签来让它正常工作。
    imgs, labels = batch

    # 在测试中我们不需要梯度，我们甚至没有标签来计算损失。
    # Using torch.no_grad() accelerates the forward process.
    with torch.no_grad():
        logits = model(imgs.to(device))

    # Take the class with greatest logit as prediction and record it.
    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

# Save predictions into the file.
with open("predicte1.csv", "w") as f:

    # The first row must be "Id, Category"
    f.write("Id,Category\n")

    # For the rest of the rows, each image id corresponds to a predicted class.
    for i, pred in  enumerate(predictions):
         f.write(f"{i},{pred}\n")
