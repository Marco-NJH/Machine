'''
DateSet 提供一起汇总方式获取其lable
DateLoad 为后面网络体质不同的数据形式
'''

from torch.utils.data import Dataset
import cv2
from PIL import Image
import os

class MyData(Dataset):

    def __init__(self,root_dir,lable_dir):
        self.root_dir = root_dir
        self.lable_dir = lable_dir
        self.path = os.path.join(self.root_dir,self.lable_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir,self.lable_dir,img_name)
        img = Image.open(img_item_path)
        lable = self.lable_dir
        return img,lable

    def __len__(self):
        return len(self.img_path)

root_dir = "com/Pytorch/Begin/hymenoptera_data/train"
ants_lable_dir = "ants"
bees_lable_dir = 'bees'
bees_Dataset = MyData(root_dir,bees_lable_dir)
ants_Dataset = MyData(root_dir,ants_lable_dir)

train_Dataset = ants_Dataset+bees_Dataset
