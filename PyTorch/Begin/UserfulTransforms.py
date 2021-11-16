from PIL import Image
from torchvision import transforms
from tensorboardX import SummaryWriter

writer = SummaryWriter("logs")
img = Image.open("hymenoptera_data/train/bees/98391118_bdb1e80cce.jpg")
print(img)
# Totensor 使用
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)

writer.add_image("Totensor",img_tensor)

#Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Norm" ,img_norm)

#resize

#

writer.close()