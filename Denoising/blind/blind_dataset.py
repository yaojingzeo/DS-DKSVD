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


class SubImagesDataset(Dataset):
    def __init__(self, root_dir, image_names, sub_image_size, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            image_names (list): List of the images names.
            sub_image_size (integer): Width of the square sub image.
            sigma (float): Level of the noise.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.sub_image_size = sub_image_size
        self.transform = transform
        self.root_dir = root_dir
        self.image_names = image_names

        self.dataset_list = [io.imread(os.path.join(self.root_dir, name)) for name in self.image_names]

        w, h = np.shape(self.dataset_list[0])
        self.number_sub_images = int(
            (w - sub_image_size + 1) * (h - sub_image_size + 1)
        )

        self.number_images = len(self.image_names)

    @staticmethod
    def extract_sub_image_from_image(image, sub_image_size, idx_sub_image):
        w, h = np.shape(image)
        w_idx, h_idx = np.unravel_index(idx_sub_image, (int(w - sub_image_size + 1), int(h - sub_image_size + 1)))
        sub_image = image[w_idx: w_idx + sub_image_size, h_idx: h_idx + sub_image_size]
        sub_image = sub_image.reshape(1, sub_image_size, sub_image_size)
        return sub_image

    def __len__(self):
        return self.number_images * self.number_sub_images

    def __getitem__(self, idx):
        idx_im, idx_sub_image = np.unravel_index(idx, (self.number_images, self.number_sub_images))

        image = self.dataset_list[idx_im]
        sub_image = self.extract_sub_image_from_image(image, self.sub_image_size, idx_sub_image)

        np.random.seed(idx)
        noise = np.random.randn(self.sub_image_size, self.sub_image_size)
        sigma_range = [0, 50]
        sigma = random.uniform(sigma_range[0], sigma_range[1])
        sub_image_noise = sub_image + sigma * noise

        if self.transform:
            sub_image = self.transform(sub_image)
            sub_image_noise = self.transform(sub_image_noise)

        return sub_image.float(), sub_image_noise.float()


class FullImagesDataset(Dataset):
    def __init__(self, root_dir: str, image_names: List[str], sigma: float, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            image_names (list): List of the name of the images.
            sigma (float): Level of the noise.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.transform = transform
        self.root_dir = root_dir
        self.sigma = sigma
        self.image_names = image_names

        self.dataset_list = [io.imread(os.path.join(self.root_dir, name)) for name in self.image_names]
        self.dataset_list_noise = [self._add_noise_to_image(np_im, k + 1e7) for (k, np_im) in
                                   enumerate(self.dataset_list)]

        self.number_images = len(self.image_names)

    def _add_noise_to_image(self, np_image, seed):
        w, h = np.shape(np_image)
        np.random.seed(int(seed))
        noise = np.random.randn(w, h)
        np_im_noise = np_image + self.sigma * noise
        return np_im_noise

    def __len__(self):
        return self.number_images

    def __getitem__(self, idx):
        image = self.dataset_list[idx]
        w, h = np.shape(image)
        image = image.reshape(1, w, h)
        image_noise = self.dataset_list_noise[idx]
        image_noise = image_noise.reshape(1, w, h)

        if self.transform:
            image = self.transform(image)
            image_noise = self.transform(image_noise)
        return image.float(), image_noise.float()


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