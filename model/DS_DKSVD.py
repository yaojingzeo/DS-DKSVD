"""
Implementation of the Deep K-SVD Denoising model, presented in
Deep K-SVD Denoising
M Scetbon, M Elad, P Milanfar
"""

import os
from skimage import io
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List
from model.cbam import CBAM

from model.cbam import CBAM


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
    def __init__(self, root_dir, image_names, sub_image_size, sigma, transform=None):
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
        self.sigma = sigma
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

        sub_image_noise = sub_image + self.sigma * noise

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


class DenoisingNet_MLP(torch.nn.Module):
    def __init__(
            self,
            patch_size,
            D_in,
            H_1,
            H_2,
            H_3,
            D_out_lam,
            D_out_lam_pd,
            T,
            min_v,
            max_v,
            Dict_init,
            c_init,
            # w_init,
            indice,
            device,
    ):
        super(DenoisingNet_MLP, self).__init__()
        self.patch_size = patch_size

        self.T = T
        self.min_v = min_v
        self.max_v = max_v

        # q, l = Dict_init.shape
        soft_comp = torch.zeros(256).to(device)
        Identity = torch.eye(256).to(device)

        self.soft_comp = soft_comp
        self.Identity = Identity
        self.device = device
        self.indice = indice

        self.linears1 = torch.nn.Linear(64, 128, bias=True).to(device)
        self.linears2 = torch.nn.Linear(128, 256, bias=True).to(device)
        self.linears3 = torch.nn.Linear(256, 128, bias=True).to(device)
        self.linears4 = torch.nn.Linear(128, 64, bias=True).to(device)


        self.Dict = torch.nn.Parameter(Dict_init)
        self.c = torch.nn.Parameter(c_init)
        self.unfold = torch.nn.Unfold(kernel_size=(self.patch_size, self.patch_size))
        self.unfold_1 = torch.nn.Unfold(kernel_size=(self.patch_size, self.patch_size), stride=8)

        self.linear1 = torch.nn.Linear(D_in, H_1, bias=True).to(device)
        self.linear2 = torch.nn.Linear(H_1, H_2, bias=True).to(device)
        self.linear3 = torch.nn.Linear(H_2, H_3, bias=True).to(device)
        self.linear4 = torch.nn.Linear(H_3, D_out_lam, bias=True).to(device)


        self.pd1 = torch.nn.Linear(D_in, H_1, bias=True).to(device)
        self.pd2 = torch.nn.Linear(H_1, H_2, bias=True).to(device)
        self.pd3 = torch.nn.Linear(H_2,  H_3, bias=True).to(device)
        self.pd4 = torch.nn.Linear(H_3, D_out_lam_pd, bias=True).to(device)

        self.linearw1 = torch.nn.Linear(64, 256, bias=True).to(device)
        # self.dropoutw1 = torch.nn.Dropout(0.5)
        self.linearw2 = torch.nn.Linear(256, 128, bias=True).to(device)
        # self.dropoutw2 = torch.nn.Dropout(0.5)
        self.linearw3 = torch.nn.Linear(128, 64, bias=True).to(device)
        # self.dropoutw3 = torch.nn.Dropout(0.5)
        self.sigmoid = torch.nn.Sigmoid()
        self.atten = CBAM(64,128).to(device)

    def soft_thresh(self, x, l):
        return torch.sign(x) * torch.max(torch.abs(x) - l, self.soft_comp)

    def change_shape(self, x,w,h):
        batch_size,w_2,channel = x.shape
        x_1 = x.transpose(1,2)#(N,64,256)
        x_2 = x_1.reshape((batch_size,channel,w,h))  #[N,64,16,16]
        return x_2

    def restore_shape(self, x):
        batch_size, channel, w, h = x.shape  # [N,64,16,16]
        x_1 = x.reshape((batch_size, channel, w * h))  # [N,64,256]
        # x_2 = x_1.transpose(1, 2)
        return x_1

    def forward(self, x):
        N, C, w, h = x.shape

        unfold = self.unfold(x)
        N, d, number_patches = unfold.shape
        unfold = unfold.transpose(1, 2)

        # —————————————————————————————————提取图像片————————————————————————————————————
        # 确定步长
        unfold_1 = self.unfold_1(x)
        N_1, d_1, number_patches_1 = unfold_1.shape   # (N_1,64,-)
        unfold_1 = unfold_1.transpose(1, 2)
        step = number_patches_1 // self.indice
        unfold_extract = unfold_1[:, 0:: step, :].to(self.device)
        unfold_extract = unfold_extract[:, :self.indice, :]
        unfold_extract = unfold_extract.contiguous()  # (N,256,64)

        sdict = self.linears1(unfold_extract).clamp(min=0)
        sdict = self.linears2(sdict).clamp(min=0)
        sdict = self.linears3(sdict).clamp(min=0)
        sdict = self.linears4(sdict)# (N,256,64)
        sdict = torch.nn.functional.normalize(sdict, dim=-1)

        sdict = self.change_shape(sdict,8,14)#(N,64,16,16)
        sdict = self.atten(sdict)#(N,64,16,16)
        sdict = self.restore_shape(sdict)#[N,64,256]

        Dict_concat = torch.cat((self.Dict.repeat(N, 1, 1),sdict),dim=-1)

        # sdict = sdict.transpose(-1, -2)

        lin = self.linear1(unfold).clamp(min=0)
        lin = self.linear2(lin).clamp(min=0)
        lin = self.linear3(lin).clamp(min=0)
        lam = self.linear4(lin)

        l = lam / self.c
        # l = l.repeat(1, 1, 112)

        lin = self.pd1(unfold).clamp(min=0)
        lin = self.pd2(lin).clamp(min=0)
        lin = self.pd3(lin).clamp(min=0)
        lampd = self.pd4(lin)

        lpd = lampd / self.c
        # lpd = lpd.repeat(1, 1, 144)

        l = torch.cat((lpd, l), dim=-1)

        y = torch.matmul(unfold, Dict_concat)
        # S = self.Identity - (1 / self.c) * self.Dict.t().mm(self.Dict)
        S = self.Identity - (1 / self.c) * Dict_concat.transpose(-1, -2) @ (Dict_concat)
        S = S.transpose(-1, -2)

        z = self.soft_thresh(y, l)
        for t in range(self.T):
            z = self.soft_thresh(torch.matmul(z, S) + (1 / self.c) * y, l)

        W = self.linearw1(unfold).clamp(min=0)
        # W = self.dropoutw1(W)
        W = self.linearw2(W).clamp(min=0)
        # W = self.dropoutw2(W)
        W = self.linearw3(W)
        W = self.sigmoid(W)

        x_pred = torch.matmul(z, Dict_concat.transpose(-1, -2))
        x_pred = torch.clamp(x_pred, min=self.min_v, max=self.max_v)
        x_pred = W * x_pred
        x_pred = x_pred.transpose(1, 2)

        normalize = torch.ones(N, number_patches, d)
        normalize = normalize.to(self.device)
        normalize = W * normalize
        normalize = normalize.transpose(1, 2)

        fold = torch.nn.functional.fold(
            x_pred, output_size=(w, h), kernel_size=(self.patch_size, self.patch_size)
        )

        norm = torch.nn.functional.fold(
            normalize,
            output_size=(w, h),
            kernel_size=(self.patch_size, self.patch_size),
        )

        res = fold / norm

        return res


class DenoisingNet_MLP_2(torch.nn.Module):
    def __init__(
            self,
            patch_size,
            D_in,
            H_1,
            H_2,
            H_3,
            D_out_lam,
            T,
            min_v,
            max_v,
            Dict_init,
            c_init,
            w_1_init,
            w_2_init,
            device,
    ):

        super(DenoisingNet_MLP_2, self).__init__()
        self.patch_size = patch_size

        self.T = T
        self.min_v = min_v
        self.max_v = max_v

        q, l = Dict_init.shape
        soft_comp = torch.zeros(l).to(device)
        Identity = torch.eye(l).to(device)
        self.soft_comp = soft_comp
        self.Identity = Identity
        self.device = device

        self.Dict = torch.nn.Parameter(Dict_init)
        self.c = torch.nn.Parameter(c_init)
        self.unfold = torch.nn.Unfold(kernel_size=(self.patch_size, self.patch_size))

        #### First Stage ####
        self.linear1 = torch.nn.Linear(D_in, H_1, bias=True)
        self.linear2 = torch.nn.Linear(H_1, H_2, bias=True)
        self.linear3 = torch.nn.Linear(H_2, H_3, bias=True)
        self.linear4 = torch.nn.Linear(H_3, D_out_lam, bias=True)

        self.w_1 = torch.nn.Parameter(w_1_init)
        ######################

        #### Second Stage ####
        self.linear1_2 = torch.nn.Linear(D_in, H_1, bias=True)
        self.linear2_2 = torch.nn.Linear(H_1, H_2, bias=True)
        self.linear3_2 = torch.nn.Linear(H_2, H_3, bias=True)
        self.linear4_2 = torch.nn.Linear(H_3, D_out_lam, bias=True)

        self.w_2 = torch.nn.Parameter(w_2_init)
        ######################

    def soft_thresh(self, x, l):
        return torch.sign(x) * torch.max(torch.abs(x) - l, self.soft_comp)

    def forward(self, x):

        N, C, w, h = x.shape

        unfold = self.unfold(x)
        N, d, number_patches = unfold.shape

        unfold = unfold.transpose(1, 2)

        lin = self.linear1(unfold).clamp(min=0)
        lin = self.linear2(lin).clamp(min=0)
        lin = self.linear3(lin).clamp(min=0)
        lam = self.linear4(lin)

        l = lam / self.c
        y = torch.matmul(unfold, self.Dict)
        S = self.Identity - (1 / self.c) * self.Dict.t().mm(self.Dict)
        S = S.t()

        z = self.soft_thresh(y, l)
        for t in range(self.T):
            z = self.soft_thresh(torch.matmul(z, S) + (1 / self.c) * y, l)

        x_pred = torch.matmul(z, self.Dict.t())
        x_pred = torch.clamp(x_pred, min=self.min_v, max=self.max_v)
        x_pred = self.w_1 * x_pred
        x_pred = x_pred.transpose(1, 2)

        normalize = torch.ones(N, number_patches, d)
        normalize = normalize.to(self.device)
        normalize = self.w_1 * normalize
        normalize = normalize.transpose(1, 2)

        fold = torch.nn.functional.fold(
            x_pred, output_size=(w, h), kernel_size=(self.patch_size, self.patch_size)
        )

        norm = torch.nn.functional.fold(
            normalize,
            output_size=(w, h),
            kernel_size=(self.patch_size, self.patch_size),
        )

        res = fold / norm

        ### Second Stage ###
        unfold = self.unfold(res)
        unfold = unfold.transpose(1, 2)

        lin = self.linear1_2(unfold).clamp(min=0)
        lin = self.linear2_2(lin).clamp(min=0)
        lin = self.linear3_2(lin).clamp(min=0)
        lam = self.linear4_2(lin)

        l = lam / self.c
        y = torch.matmul(unfold, self.Dict)

        z = self.soft_thresh(y, l)
        for t in range(self.T):
            z = self.soft_thresh(torch.matmul(z, S) + (1 / self.c) * y, l)

        x_pred = torch.matmul(z, self.Dict.t())
        x_pred = torch.clamp(x_pred, min=self.min_v, max=self.max_v)
        x_pred = self.w_2 * x_pred
        x_pred = x_pred.transpose(1, 2)

        normalize = torch.ones(N, number_patches, d)
        normalize = normalize.to(self.device)
        normalize = self.w_2 * normalize
        normalize = normalize.transpose(1, 2)

        fold = torch.nn.functional.fold(
            x_pred, output_size=(w, h), kernel_size=(self.patch_size, self.patch_size)
        )

        norm = torch.nn.functional.fold(
            normalize,
            output_size=(w, h),
            kernel_size=(self.patch_size, self.patch_size),
        )

        res = fold / norm

        return res

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class DenoisingNet_MLP_3(torch.nn.Module):
    def __init__(
            self,
            patch_size,
            D_in,
            H_1,
            H_2,
            # H_3,
            D_out_lam,
            D_out_lam_pd,
            T,
            min_v,
            max_v,
            Dict_init,
            c,
            indice,
            device,
    ):

        super(DenoisingNet_MLP_3, self).__init__()
        self.patch_size = patch_size
        self.p = patch_size * patch_size

        self.T = T
        self.min_v = min_v
        self.max_v = max_v

        D = D_out_lam+D_out_lam_pd
        soft_comp = torch.zeros(D).to(device)
        Identity = torch.eye(D).to(device)
        self.soft_comp = soft_comp
        self.Identity = Identity
        self.device = device
        self.indice = indice

        self.Dict = torch.nn.Parameter(Dict_init)
        self.c = torch.nn.Parameter(c)
        self.unfold = torch.nn.Unfold(kernel_size=(self.patch_size, self.patch_size))
        self.unfold_1 = torch.nn.Unfold(kernel_size=(self.patch_size, self.patch_size), stride=8)

        self.linears1 = torch.nn.Linear(self.p , 2*self.p , bias=True).to(device)
        self.linears2 = torch.nn.Linear(2*self.p , 4*self.p , bias=True).to(device)
        self.linears3 = torch.nn.Linear(4*self.p , 2*self.p , bias=True).to(device)
        self.linears4 = torch.nn.Linear(2*self.p , self.p , bias=True).to(device)
        self.atten = CBAM(self.p , 2*self.p ).to(device)

        #### First Stage ####
        self.linear1 = torch.nn.Linear(D_in, H_1, bias=True).to(device)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.linear2 = torch.nn.Linear(H_1, H_2, bias=True).to(device)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.linear3 = torch.nn.Linear(H_2, D_out_lam, bias=True).to(device)
        # self.dropout3 = torch.nn.Dropout(0.5)
        # self.linear4 = torch.nn.Linear(H_3, D_out_lam, bias=True).to(device)

        self.pd1 = torch.nn.Linear(D_in, H_1, bias=True).to(device)
        self.dropoutpd1 = torch.nn.Dropout(0.5)
        self.pd2 = torch.nn.Linear(H_1, 512, bias=True).to(device)
        self.dropoutpd2 = torch.nn.Dropout(0.5)
        self.pd3 = torch.nn.Linear(512,  D_out_lam_pd, bias=True).to(device)
        # self.dropoutpd3 = torch.nn.Dropout(0.5)
        # self.pd4 = torch.nn.Linear(H_3, D_out_lam_pd, bias=True).to(device)

        #####################

        ### Second Stage ####
        self.linear1_2 = torch.nn.Linear(D_in, H_1, bias=True).to(device)
        self.dropout1_2 = torch.nn.Dropout(0.5)
        self.linear2_2 = torch.nn.Linear(H_1, H_2, bias=True).to(device)
        self.dropout2_2 = torch.nn.Dropout(0.5)
        self.linear3_2 = torch.nn.Linear(H_2, D_out_lam, bias=True).to(device)
        # self.dropout3_2 = torch.nn.Dropout(0.5)
        # self.linear4_2 = torch.nn.Linear(H_3, D_out_lam, bias=True).to(device)

        self.pd1_2 = torch.nn.Linear(D_in, H_1, bias=True).to(device)
        self.dropoutpd1_2 = torch.nn.Dropout(0.5)
        self.pd2_2 = torch.nn.Linear(H_1, 512, bias=True).to(device)
        self.dropoutpd2_2 = torch.nn.Dropout(0.5)
        self.pd3_2 = torch.nn.Linear(512, D_out_lam_pd, bias=True).to(device)
        # self.dropoutpd3_2 = torch.nn.Dropout(0.5)
        # self.pd4_2 = torch.nn.Linear(H_3, D_out_lam_pd, bias=True).to(device)

        ######################

        #### Third Stage ####
        # self.linear1_3 = torch.nn.Linear(D_in, H_1, bias=True).to(device)
        # self.linear2_3 = torch.nn.Linear(H_1, H_2, bias=True).to(device)
        # self.linear3_3 = torch.nn.Linear(H_2, H_3, bias=True).to(device)
        # self.linear4_3 = torch.nn.Linear(H_3, D_out_lam, bias=True).to(device)
        #
        # self.pd1_3 = torch.nn.Linear(D_in, H_1, bias=True).to(device)
        # self.pd2_3 = torch.nn.Linear(H_1, H_2, bias=True).to(device)
        # self.pd3_3= torch.nn.Linear(H_2, H_3, bias=True).to(device)
        # self.pd4_3 = torch.nn.Linear(H_3, D_out_lam_pd, bias=True).to(device)


        ######################


        self.linearw1 = torch.nn.Linear(256, 1024, bias=True).to(device)
        self.dropoutw1 = torch.nn.Dropout(0.5)
        self.linearw2 = torch.nn.Linear(1024, 512, bias=True).to(device)
        self.dropoutw2 = torch.nn.Dropout(0.5)
        self.linearw3 = torch.nn.Linear(512, 256, bias=True).to(device)
        self.dropoutw3 = torch.nn.Dropout(0.5)
        self.sigmoid = torch.nn.Sigmoid()


    def soft_thresh(self, x, l):
        return torch.sign(x) * torch.max(torch.abs(x) - l, self.soft_comp)


    def change_shape(self, x,w,h):
        batch_size,w_2,channel = x.shape
        x_1 = x.transpose(1,2)#(N,64,256)
        x_2 = x_1.reshape((batch_size,channel,w,h))  #[N,64,16,16]
        return x_2

    def restore_shape(self, x):
        batch_size, channel, w, h = x.shape  # [N,64,16,16]
        x_1 = x.reshape((batch_size, channel, w * h))  # [N,64,256]
        # x_2 = x_1.transpose(1, 2)
        return x_1

    def forward(self, x):
        N, C, w, h = x.shape

        unfold = self.unfold(x)
        N, d, number_patches = unfold.shape
        unfold = unfold.transpose(1, 2)

        # —————————————————————————————————提取图像片————————————————————————————————————
        # 确定步长
        unfold_1 = self.unfold_1(x)
        N_1, d_1, number_patches_1 = unfold_1.shape  # (N_1,64,-)
        unfold_1 = unfold_1.transpose(1, 2)
        step = number_patches_1 // self.indice
        unfold_extract = unfold_1[:, 0:: step, :]
        unfold_extract = unfold_extract[:, :self.indice, :]
        unfold_extract = unfold_extract.contiguous()  # (N,256,64)


        sdict = self.linears1(unfold_extract).clamp(min=0)
        sdict = self.linears2(sdict).clamp(min=0)
        sdict = self.linears3(sdict).clamp(min=0)
        sdict = self.linears4(sdict)  # (N,256,64)
        sdict = torch.nn.functional.normalize(sdict, dim=-1)

        sdict = self.change_shape(sdict, 8, 14)  # (N,64,16,16)
        sdict = self.atten(sdict)  # (N,64,16,16)
        sdict = self.restore_shape(sdict)  # [N,64,256]

        Dict_concat = torch.cat((self.Dict.repeat(N, 1, 1), sdict), dim=-1)

        # sdict = sdict.transpose(-1, -2)

        lin = self.linear1(unfold).clamp(min=0)
        lin = self.dropout1(lin)
        lin = self.linear2(lin).clamp(min=0)
        lin = self.dropout2(lin)
        # lin = self.linear3(lin).clamp(min=0)
        # lin = self.dropout3(lin)
        lam = self.linear3(lin)
        ls = lam / self.c

        lin = self.pd1(unfold).clamp(min=0)
        lin = self.dropoutpd1(lin)
        lin = self.pd2(lin).clamp(min=0)
        lin = self.dropoutpd2(lin)
        # lin = self.pd3(lin).clamp(min=0)
        # lin = self.dropoutpd3(lin)
        lampd = self.pd3(lin)
        lpd = lampd / self.c

        l = torch.cat((lpd, ls), dim=-1)

        y = torch.matmul(unfold, Dict_concat)
        S = self.Identity - (1 / self.c) * Dict_concat.transpose(-1, -2) @ (Dict_concat)
        S = S.transpose(-1, -2)

        z = self.soft_thresh(y, l)
        for t in range(self.T):
            z = self.soft_thresh(torch.matmul(z, S) + (1 / self.c) * y, l)

        W = self.linearw1(unfold).clamp(min=0)
        W = self.dropoutw1(W)
        W = self.linearw2(W).clamp(min=0)
        W = self.dropoutw2(W)
        W = self.linearw3(W)
        W = self.sigmoid(W)

        x_pred = torch.matmul(z, Dict_concat.transpose(-1, -2))
        x_pred = torch.clamp(x_pred, min=self.min_v, max=self.max_v)
        x_pred = W * x_pred
        x_pred = x_pred.transpose(1, 2)

        normalize = torch.ones(N, number_patches, d)
        normalize = normalize.to(self.device)
        normalize = W * normalize
        normalize = normalize.transpose(1, 2)

        fold = torch.nn.functional.fold(
            x_pred, output_size=(w, h), kernel_size=(self.patch_size, self.patch_size)
        )

        norm = torch.nn.functional.fold(
            normalize,
            output_size=(w, h),
            kernel_size=(self.patch_size, self.patch_size),
        )

        res = fold / norm

        ## Second Stage ###
        unfold = self.unfold(res)
        unfold = unfold.transpose(1, 2)

        # —————————————————————————————————提取图像片————————————————————————————————————
        # 确定步长
        unfold_1 = self.unfold_1(res)
        N_1, d_1, number_patches_1 = unfold_1.shape  # (N_1,64,-)
        unfold_1 = unfold_1.transpose(1, 2)
        step = number_patches_1 // self.indice
        unfold_extract = unfold_1[:, 0:: step, :]
        unfold_extract = unfold_extract[:, :self.indice, :]
        unfold_extract = unfold_extract.contiguous()  # (N,256,64)

        sdict = self.linears1(unfold_extract).clamp(min=0)
        sdict = self.linears2(sdict).clamp(min=0)
        sdict = self.linears3(sdict).clamp(min=0)
        sdict = self.linears4(sdict)  # (N,256,64)
        sdict = torch.nn.functional.normalize(sdict, dim=-1)

        sdict = self.change_shape(sdict, 8, 14)  # (N,64,16,16)
        sdict = self.atten(sdict)  # (N,64,16,16)
        sdict = self.restore_shape(sdict)  # [N,64,256]

        Dict_concat = torch.cat((self.Dict.repeat(N, 1, 1), sdict), dim=-1)


        lin = self.linear1_2(unfold).clamp(min=0)
        lin = self.dropout1_2(lin)
        lin = self.linear2_2(lin).clamp(min=0)
        lin = self.dropout2_2(lin)
        # lin = self.linear3_2(lin).clamp(min=0)
        # lin = self.dropout3_2(lin)
        lam = self.linear3_2(lin).to(self.device)
        ls = lam / self.c

        lin = self.pd1_2(unfold).clamp(min=0)
        lin = self.dropoutpd1_2(lin)
        lin = self.pd2_2(lin).clamp(min=0)
        lin = self.dropoutpd2_2(lin)
        # lin = self.pd3_2(lin).clamp(min=0)
        # lin = self.dropoutpd3_2(lin)
        lampd = self.pd3_2(lin)
        lpd = lampd / self.c

        l = torch.cat((lpd, ls), dim=-1)
        y = torch.matmul(unfold, Dict_concat)
        S = self.Identity - (1 / self.c) * Dict_concat.transpose(-1, -2) @ (Dict_concat)
        S = S.transpose(-1, -2)

        z = self.soft_thresh(y, l)
        for t in range(self.T):
            z = self.soft_thresh(torch.matmul(z, S) + (1 / self.c)* y, l)

        W = self.linearw1(unfold).clamp(min=0)
        W = self.dropoutw1(W)
        W = self.linearw2(W).clamp(min=0)
        W = self.dropoutw2(W)
        W = self.linearw3(W)
        W = self.sigmoid(W)

        x_pred = torch.matmul(z, Dict_concat.transpose(-1, -2))
        x_pred = torch.clamp(x_pred, min=self.min_v, max=self.max_v)
        x_pred = W * x_pred
        x_pred = x_pred.transpose(1, 2)

        normalize = torch.ones(N, number_patches, d)
        normalize = normalize.to(self.device)
        normalize = W * normalize
        normalize = normalize.transpose(1, 2)

        fold = torch.nn.functional.fold(
            x_pred, output_size=(w, h), kernel_size=(self.patch_size, self.patch_size)
        )

        norm = torch.nn.functional.fold(
            normalize,
            output_size=(w, h),
            kernel_size=(self.patch_size, self.patch_size),
        )

        res = fold / norm

        # ### Third Stage ###
        # unfold = self.unfold(res)
        # unfold = unfold.transpose(1, 2)
        #
        # # —————————————————————————————————提取图像片————————————————————————————————————
        # # 确定步长
        # unfold_1 = self.unfold_1(res)
        # N_1, d_1, number_patches_1 = unfold_1.shape  # (N_1,64,-)
        # unfold_1 = unfold_1.transpose(1, 2)
        # step = number_patches_1 // self.indice
        # unfold_extract = unfold_1[:, 0:: step, :]
        # unfold_extract = unfold_extract[:, :self.indice, :]
        # unfold_extract = unfold_extract.contiguous()  # (N,256,64)
        #
        # sdict = self.linears1(unfold_extract).clamp(min=0)
        # sdict = self.linears2(sdict).clamp(min=0)
        # sdict = self.linears3(sdict).clamp(min=0)
        # sdict = self.linears4(sdict)  # (N,256,64)
        # sdict = torch.nn.functional.normalize(sdict, dim=-1)
        #
        # sdict = self.change_shape(sdict, 8, 14)  # (N,64,16,16)
        # sdict = self.atten(sdict)  # (N,64,16,16)
        # sdict = self.restore_shape(sdict)  # [N,64,256]
        #
        # Dict_concat = torch.cat((self.Dict.repeat(N, 1, 1), sdict), dim=-1)
        #
        #
        # lin = self.linear1_3(unfold).clamp(min=0)
        # lin = self.linear2_3(lin).clamp(min=0)
        # lin = self.linear3_3(lin).clamp(min=0)
        # lam = self.linear4_3(lin).to(self.device)
        # ls = lam / self.c
        #
        # lin = self.pd1_3(unfold).clamp(min=0)
        # lin = self.pd2_3(lin).clamp(min=0)
        # lin = self.pd3_3(lin).clamp(min=0)
        # lampd = self.pd4_3(lin)
        # lpd = lampd / self.c
        #
        #
        # l = torch.cat((lpd, ls), dim=-1)
        # y = torch.matmul(unfold, Dict_concat)
        # S = self.Identity - (1 / self.c) * Dict_concat.transpose(-1, -2) @ (Dict_concat)
        # S = S.transpose(-1, -2)
        #
        # z = self.soft_thresh(y, l)
        # for t in range(self.T):
        #     z = self.soft_thresh(torch.matmul(z, S) + (1 / self.c) * y, l)
        #
        # W = self.linearw1(unfold).clamp(min=0)
        # # W = self.dropoutw1(W)
        # W = self.linearw2(W).clamp(min=0)
        # # W = self.dropoutw2(W)
        # W = self.linearw3(W)
        # W = self.sigmoid(W)
        #
        #
        # x_pred = torch.matmul(z, Dict_concat.transpose(-1, -2))
        # x_pred = torch.clamp(x_pred, min=self.min_v, max=self.max_v)
        # x_pred = W * x_pred
        # x_pred = x_pred.transpose(1, 2)
        #
        # normalize = torch.ones(N, number_patches, d)
        # normalize = normalize.to(self.device)
        # normalize = W * normalize
        # normalize = normalize.transpose(1, 2)
        #
        # fold = torch.nn.functional.fold(
        #     x_pred, output_size=(w, h), kernel_size=(self.patch_size, self.patch_size)
        # )
        #
        # norm = torch.nn.functional.fold(
        #     normalize,
        #     output_size=(w, h),
        #     kernel_size=(self.patch_size, self.patch_size),
        # )
        #
        # res = fold / norm

        return res
