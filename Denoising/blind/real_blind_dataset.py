import os

from scipy import linalg
from skimage import io
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List
from torch import nn
from typing import List
import time
from torch.cuda import Event, Stream
import random
import time
import numbers
import pdb
# device =torch.device("cuda:0")
def order_F_to_C(n):
    idx = np.arange(0, n ** 2)
    idx = idx.reshape(n, n, order="F")
    idx = idx.reshape(n ** 2, order="C")
    idx = list(idx)
    return idx


def init_dct(n, m):
    """ Compute the Overcomplete Discrete Cosinus Transform. """
    oc_dictionary = np.zeros((n, m))
    for k in range(m):
        V = np.cos(np.arange(0, n) * k * np.pi / m)
        if k > 0:
            V = V - np.mean(V)
        oc_dictionary[:, k] = V / np.linalg.norm(V)
    oc_dictionary = np.kron(oc_dictionary, oc_dictionary)
    oc_dictionary = oc_dictionary.dot(np.diag(1 / np.sqrt(np.sum(oc_dictionary ** 2, axis=0))))
    idx = np.arange(0, n ** 2)
    idx = idx.reshape(n, n, order="F")
    idx = idx.reshape(n ** 2, order="C")
    oc_dictionary = oc_dictionary[idx, :]
    oc_dictionary = torch.from_numpy(oc_dictionary).float()
    return oc_dictionary



# class SubImagesDataset(Dataset):
#     def __init__(self, noisy_dir, clean_dir, noisy_names, clean_names, sub_image_size, transform=None):
#         """
#         Args:
#             noisy_dir (string): 目录，包含带噪声的图像。
#             clean_dir (string): 目录，包含对应的无噪声图像。
#             noisy_names (list): 带噪声的图像文件名列表。
#             clean_names (list): 真实无噪声图像的文件名列表。
#             sub_image_size (int): 需要裁剪的子图像大小。
#             transform (callable, optional): 数据变换操作。
#         """
#         self.sub_image_size = sub_image_size
#         self.transform = transform
#         self.noisy_dir = noisy_dir
#         self.clean_dir = clean_dir
#         self.noisy_names = noisy_names
#         self.clean_names = clean_names
#
#         # 确保 noisy_names 和 clean_names 数量相同
#         assert len(noisy_names) == len(clean_names), "噪声图像和真实图像的文件数不匹配！"
#
#         # 读取所有图像
#         self.noisy_images = [io.imread(os.path.join(self.noisy_dir, name)) for name in self.noisy_names]
#         self.clean_images = [io.imread(os.path.join(self.clean_dir, name)) for name in self.clean_names]
#
#         # 计算每张图像可以裁剪出的子图个数
#         w, h = self.noisy_images[0].shape[:2]
#         self.number_sub_images = (w - sub_image_size + 1) * (h - sub_image_size + 1)
#         self.number_images = len(self.noisy_names)
#
#     @staticmethod
#     def extract_sub_image(image, sub_image_size, idx_sub_image):
#         """从完整图像中裁剪出子图像"""
#         w, h = image.shape[:2]
#         w_idx, h_idx = np.unravel_index(idx_sub_image, (w - sub_image_size + 1, h - sub_image_size + 1))
#         return image[w_idx: w_idx + sub_image_size, h_idx: h_idx + sub_image_size]
#
#     def __len__(self):
#         return self.number_images * self.number_sub_images
#
#     def __getitem__(self, idx):
#         """返回一对 (噪声子图, 真实子图)"""
#         # 确定是哪张图像，以及在该图像中的裁剪索引
#         idx_im, idx_sub_image = np.unravel_index(idx, (self.number_images, self.number_sub_images))
#
#         noisy_image = self.noisy_images[idx_im]
#         clean_image = self.clean_images[idx_im]
#
#         # 以相同的方式裁剪噪声图像和真实图像
#         noisy_sub_image = self.extract_sub_image(noisy_image, self.sub_image_size, idx_sub_image)
#         clean_sub_image = self.extract_sub_image(clean_image, self.sub_image_size, idx_sub_image)
#         # 查看取值范围
#
#
#         # 转换为 PyTorch 张量，并执行数据增强
#         if self.transform:
#             noisy_sub_image = self.transform(noisy_sub_image)
#             clean_sub_image = self.transform(clean_sub_image)
#
#         noisy_sub_image = noisy_sub_image.unsqueeze(0)
#         clean_sub_image = clean_sub_image.unsqueeze(0)
#         # clean_min = clean_sub_image.min().item()
#         # clean_max = clean_sub_image.max().item()
#         # noisy_min = noisy_sub_image.min().item()
#         # noisy_max = noisy_sub_image.max().item()
#         #
#         # print(f"真实图像取值范围: 最小值 = {clean_min}, 最大值 = {clean_max}")
#         # print(f"噪声图像取值范围: 最小值 = {noisy_min}, 最大值 = {noisy_max}")
#
#         return  clean_sub_image.float(), noisy_sub_image.float()

class SubImagesDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, noisy_names, clean_names, sub_image_size, stride, transform=None):
        """
        Args:
            noisy_dir (string): 目录，包含带噪声的图像。
            clean_dir (string): 目录，包含对应的无噪声图像。
            noisy_names (list): 带噪声的图像文件名列表。
            clean_names (list): 真实无噪声图像的文件名列表。
            sub_image_size (int): 需要裁剪的子图像大小。
            stride (int): 采样步长，控制子图像之间的间隔。
            transform (callable, optional): 数据变换操作。
        """
        self.sub_image_size = sub_image_size
        self.stride = stride  # 添加步长参数
        self.transform = transform
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.noisy_names = noisy_names
        self.clean_names = clean_names

        # 确保 noisy_names 和 clean_names 数量相同
        assert len(noisy_names) == len(clean_names), "噪声图像和真实图像的文件数不匹配！"

        # 读取所有图像
        self.noisy_images = [io.imread(os.path.join(self.noisy_dir, name)) for name in self.noisy_names]
        self.clean_images = [io.imread(os.path.join(self.clean_dir, name)) for name in self.clean_names]

        # 计算每张图像可以裁剪出的子图个数，考虑步长
        w, h = self.noisy_images[0].shape[:2]
        self.number_sub_images = ((w - sub_image_size) // stride + 1) * ((h - sub_image_size) // stride + 1)
        self.number_images = len(self.noisy_names)

    def extract_sub_image(self, image, idx_sub_image):
        """从完整图像中裁剪出子图像，考虑步长"""
        w, h = image.shape[:2]
        grid_w = (w - self.sub_image_size) // self.stride + 1
        w_idx, h_idx = np.unravel_index(idx_sub_image, (grid_w, (h - self.sub_image_size) // self.stride + 1))

        # 计算实际的起始位置，考虑步长
        w_start = w_idx * self.stride
        h_start = h_idx * self.stride

        return image[w_start: w_start + self.sub_image_size, h_start: h_start + self.sub_image_size]

    def __len__(self):
        return self.number_images * self.number_sub_images

    def __getitem__(self, idx):
        """返回一对 (噪声子图, 真实子图)"""
        # 确定是哪张图像，以及在该图像中的裁剪索引
        idx_im, idx_sub_image = np.unravel_index(idx, (self.number_images, self.number_sub_images))

        noisy_image = self.noisy_images[idx_im]
        clean_image = self.clean_images[idx_im]

        # 以相同的方式裁剪噪声图像和真实图像
        noisy_sub_image = self.extract_sub_image(noisy_image, idx_sub_image)
        clean_sub_image = self.extract_sub_image(clean_image, idx_sub_image)

        # 转换为 PyTorch 张量，并执行数据增强
        if self.transform:
            noisy_sub_image = self.transform(noisy_sub_image)
            clean_sub_image = self.transform(clean_sub_image)

        noisy_sub_image = noisy_sub_image.unsqueeze(0)
        clean_sub_image = clean_sub_image.unsqueeze(0)

        return clean_sub_image.float(), noisy_sub_image.float()






class FullImagesDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir,  clean_names: List[str], noisy_names: List[str], transform=None):
        """
        Args:
            root_dir (string): 包含所有图像的目录。
            clean_names (list): 真实无噪声图像的文件名列表。
            noisy_names (list): 带噪声的图像文件名列表。
            transform (callable, optional): 预处理或数据增强操作。
        """

        self.transform = transform
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.clean_names = clean_names
        self.noisy_names = noisy_names

        assert len(clean_names) == len(noisy_names), "真实图像和噪声图像的数量不匹配！"

        # 读取所有图像
        self.noisy_images = [io.imread(os.path.join(self.noisy_dir, name)) for name in self.noisy_names]
        self.clean_images = [io.imread(os.path.join(self.clean_dir, name)) for name in self.clean_names]


        self.number_images = len(self.clean_names)

    def __len__(self):
        return self.number_images

    def __getitem__(self, idx):
        """返回一对 (真实图像, 噪声图像)"""
        clean_image = self.clean_images[idx]
        noisy_image = self.noisy_images[idx]

        # 调整维度 [H, W] -> [1, H, W]
        w, h = clean_image.shape
        clean_image = clean_image.reshape(1, w, h)
        noisy_image = noisy_image.reshape(1, w, h)

        # 应用数据变换
        if self.transform:
            clean_image = self.transform(clean_image)
            noisy_image = self.transform(noisy_image)

        return clean_image.float(), noisy_image.float()




class ToTensor(object):
    """ Convert ndarrays to Tensors. """

    def __call__(self, image):
        return torch.from_numpy(image)


class Normalize(object):
    """ Normalize the images. """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        return (image - self.mean) / self.std