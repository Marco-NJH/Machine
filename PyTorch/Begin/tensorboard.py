# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
img_path = "hymenoptera_data/train/bees/16838648_415acd9e3f.jpg"
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)
writer.add_image("test",img_array,1,dataformats="HWC")
for i in range(100):
    writer.add_scalar("y=2x",2*i,i)

writer.close()

#tensorboard --logdir PyTorch/Begin/logs --host=127.0.0.1