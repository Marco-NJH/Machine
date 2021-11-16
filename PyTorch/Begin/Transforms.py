from PIL import Image
from tensorboardX import SummaryWriter
from torchvision import transforms

img_path = "hymenoptera_data/train/ants/5650366_e22b7e1065.jpg"
img = Image.open(img_path)

write = SummaryWriter("logs")

# 如何使用transforms
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
# print(tensor_img)
write.add_image("Tensor_img",tensor_img)