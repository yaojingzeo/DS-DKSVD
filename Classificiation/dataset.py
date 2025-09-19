import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 定义Dataset类
class TumorDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        """
        Args:
            image_folder (str): 图像文件夹路径
            transform (callable, optional): 需要应用于图像的变换
        """
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = []
        # self.sigma=sigma
        self.labels = []
        self.num_abnormal_images = 0

        # 遍历文件夹中的所有图像文件
        for filename in os.listdir(image_folder):
            if filename.endswith('.jpg'):
                # 获取图像路径
                image_path = os.path.join(image_folder, filename)
                self.image_paths.append(image_path)

                # 标签化：'y'代表肿瘤，'no'代表没有肿瘤
                label = 1 if filename.startswith('y') else 0
                self.labels.append(label)

    def __len__(self):
        # 返回数据集的大小
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 根据索引加载图像
        image_path = self.image_paths[idx]
        # 读取图像并转换为灰度图（如果需要灰度图像）
        image = Image.open(image_path).convert('L')  # 转换为灰度图（1通道）
        if self.transform:
            image = self.transform(image)

        # 获取标签
        label = self.labels[idx]


        return image, label, os.path.basename(image_path)



def get_train_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小为128x128
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(15),  # 随机旋转
        transforms.ToTensor(),  # 转换为Tensor格式
        transforms.Normalize(mean=[0.5], std=[0.5])  # 将 [0, 1] 映射到 [-1, 1]
    ])

def get_test_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小为128x128
        transforms.ToTensor(),  # 转换为Tensor格式
        transforms.Normalize(mean=[0.5], std=[0.5])  # 将 [0, 1] 映射到 [-1, 1]
    ])



# train_transform = transforms.Compose([
#         transforms.Resize((128, 128)),  # 调整图像大小为256x256
#         transforms.RandomHorizontalFlip(),  # 随机水平翻转
#         transforms.RandomRotation(15),  # 随机旋转
#         transforms.ToTensor(),  # 转换为Tensor格式
#         transforms.Normalize(mean=[0.5], std=[0.5])  # 将 [0, 1] 映射到 [-1, 1]
#         # transforms.Lambda(lambda x: 2 * x - 1) # 将范围从 [0, 1] 映射到 [-1, 1]
#     ])
#
# # 定义图像预处理过程：可选的裁剪、缩放、标准化等
# test_transform = transforms.Compose([
#     transforms.Resize((128, 128)),  # 调整图像大小为256x256
#     transforms.ToTensor(),  # 转换为Tensor格式
#     transforms.Normalize(mean=[0.5], std=[0.5])
#     # transforms.Lambda(lambda x: 2 * x - 1) # 将范围从 [0, 1] 映射到 [-1, 1]
# ])
#
# test_dataset = TumorDataset(image_folder='dataset/test', sigma=50, transform=test_transform)
# dataloader_test  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
#
# train_dataset = TumorDataset(image_folder='dataset/test', sigma=50,transform=test_transform)
# dataloader_train  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
#
# for i, (image_t, image_noise, label) in enumerate(dataloader_test):
#     # 获取 image_noise 的最小值和最大值
#     min_value = image_t.min().item()
#     max_value = image_t.max().item()
#
#     # 检查是否在 [-1, 1] 范围内
#     if min_value < -1 or max_value > 1:
#         print(f"Training Image {i + 1}: min = {min_value}, max = {max_value} - Out of range!")
#     else:
#         print(f"Training Image {i + 1}: min = {min_value}, max = {max_value} - In range.")

#
# for i, (image, label) in enumerate(dataloader_test ):
#     print(f"Test Image {i+1}: min = {image.min()}, max = {image.max()}")