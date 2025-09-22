"""
"""

import numpy as np
from scipy import linalg
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from skimage.metrics import structural_similarity
import time

import os
import re
from skimage.metrics import structural_similarity as ssim

from model import DS_DKSVD

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# Overcomplete Discrete Cosinus Transform:
patch_size = 16
m = 20
Dict_init = DS_DKSVD.init_dct(patch_size, m)
Dict_init = Dict_init.to(device)

# Squared Spectral norm:
c_init = linalg.norm(Dict_init.cpu(), ord=2) ** 2
c_init = torch.FloatTensor((c_init,))
c_init = c_init.to(device)

# Average weight:
w_init1 = torch.normal(mean=1, std=1 / 10 * torch.ones(patch_size ** 2)).float()
w_init2 = torch.normal(mean=1, std=1 / 10 * torch.ones(patch_size ** 2)).float()
w_init1 = w_init1.to(device)
w_init2 = w_init2.to(device)

def InNormalization(image):
    mean=255/2
    std=255/2
    return image*std+mean

# log_dir_param= "model_247.pth"
# checkpoint = torch.load(log_dir_param, map_location=device)
# loaded_dict = checkpoint['model_state_dict']
#
# # 提取需要的参数
# Dict_init_loaded = loaded_dict['module.Dict']
# # print(Dict_init_loaded)
# D_in, H_1, H_2, H_3, D_out_lam, T, min_v, max_v = patch_size ** 2, 512, 256, 128, 1, 7, -1, 1
# D_in, H_1, H_2, H_3, D_out_lam, D_out_lam_pd,T, min_v, max_v, indice = patch_size ** 2, 128, 256, 128, 112, 144, 7, -1, 1,112
D_in, H_1, H_2,  D_out_lam, D_out_lam_pd,T, min_v, max_v, indice = patch_size ** 2, 512, 768, 112, 400, 7, -1, 1,112
model = DS_DKSVD.DenoisingNet_MLP_3(
    patch_size,
    D_in,
    H_1,
    H_2,
    D_out_lam,
    D_out_lam_pd,
    T,
    min_v,
    max_v,
    Dict_init,
    c_init,
    # w_init1,
    # w_init2,
    # w_init,
    indice,
    device,
)
# ———————————————————————————————导入参数——————————————————————————————
def extract_number(filename):
    """从文件名中提取数字"""
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    return -1


model_params_folder = "PATH"
files = sorted(os.listdir(model_params_folder), key=extract_number)
# 寻找中断点
start_index = 0
for i, model_file in enumerate(files):
    if model_file.startswith("model"):
        start_index = i
        break
for model_file in files[start_index:]:
    print(model_file)
    if model_file.endswith(".pth"):

        model_path = os.path.join(model_params_folder, model_file)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        iter = checkpoint['iter']
        epoch = checkpoint['epoch']
        print('加载 epoch {} 成功！'.format(epoch))
        print('加载 i {} 成功！'.format(iter))
        loaded_dict_1 = checkpoint['model_state_dict']

        # 打印不匹配的键以进行调试
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(loaded_dict_1.keys())
        missing_keys = model_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - model_keys

        if missing_keys:
            print("缺失的键:", missing_keys)
        if unexpected_keys:
            print("意外的键:", unexpected_keys)

        # 如果检查点键中存在 "module." 前缀，则移除
        new_loaded_dict_1 = {}
        for key, value in loaded_dict_1.items():
            new_key = key.replace("module.", "")
            new_loaded_dict_1[new_key] = value

        # 创建一个新的状态字典，并对齐键
        state_dict = model.state_dict()
        state_dict.update({k: v for k, v in new_loaded_dict_1.items() if k in state_dict})

        # 加载模型参数
        model.load_state_dict(state_dict)

        print('模型参数加载成功！')


        #测试集
        datasets = [
            # "test_set12",
            # "test_gray",
            "Urban100",
            # "depth",

        ]
        # ————————————————————————————————————————————————————————————————————
        for dataset in datasets:
            # Test image names:
            start=time.time()
            file_test_path1 = "d_Test_txt"
            file_test_path = os.path.join(file_test_path1, f"{dataset}.txt")
            # print(file_test_path)
            data = "real_dataset"
            data_path = os.path.join(data , f"{dataset}")
            print(data_path)
            file_test = open(file_test_path, "r")
            onlyfiles_test = [e.strip() for e in file_test]
            file_test.close()

            # Rescaling in [-1, 1]:
            mean = 255 / 2
            std = 255 / 2
            data_transform = transforms.Compose(
                [DS_DKSVD.Normalize(mean=mean, std=std), DS_DKSVD.ToTensor()]
            )
            #———————————————————————————————————————————————————— Noise level:————————————————————————————————————————————————————————
            sigma=15
            # Test Dataset:
            my_Data_test = DS_DKSVD.FullImagesDataset(
                root_dir=data_path, image_names=onlyfiles_test, sigma=sigma, transform=data_transform
            )

            dataloader_test = DataLoader(my_Data_test, batch_size=1, shuffle=False, num_workers=0)

            # Results folder:
            results_folder = "TEST"
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)

            # Create file path:
            file_path = os.path.join(results_folder, f"{dataset}_{model_file}.csv")
            # Open file and write device info
            file_to_print = open(file_path, "w")
            file_to_print.write(str(device) + "\n")
            file_to_print.flush()

            with torch.no_grad():
                model.eval()
                list_PSNR = []
                list_PSNR_init = []
                list_SSIM = []
                file_to_print.write("File,Init_PSNR,Test_PSNR\n")

                for k, (image_true, image_noise) in enumerate(dataloader_test, 0):
                    image_true_t = image_true[0, 0, :, :].to(device)
                    image_noise_0 = image_noise[0, 0, :, :].to(device)
                    image_noise_t = image_noise.to(device)
                    image_restored_t = model(image_noise_t)[0, 0, :, :]

                    PSNR_init = 10 * torch.log10(4 / torch.mean((image_true_t - image_noise_0) ** 2))
                    file_to_print.write(f"{onlyfiles_test[k]},{PSNR_init},{''}\n")
                    file_to_print.flush()

                    list_PSNR_init.append(PSNR_init.cpu().numpy())

                    # Calculate PSNR
                    PSNR = 10 * torch.log10(4 / torch.mean((image_true_t - image_restored_t) ** 2)).cpu()
                    file_to_print.write(f"{onlyfiles_test[k]},,{PSNR}\n")
                    file_to_print.flush()
                    list_PSNR.append(PSNR)

                    # Calculate SSIM
                    img1 = InNormalization(image_true_t.cpu()).numpy().astype('uint8')
                    img_noise1 = InNormalization(image_noise_0.cpu()).numpy().astype('uint8')
                    real_denoise = InNormalization(image_restored_t.cpu()).numpy().astype('uint8')
                    ssim_value = structural_similarity(img1, real_denoise, data_range=255)
                    file_to_print.write(f"{onlyfiles_test[k]},{ssim_value},{''}\n")
                    file_to_print.flush()
                    list_SSIM.append(ssim_value)

                    print(
                        f"Image {k + 1}/{len(dataloader_test)} - Dataset: {dataset} - File: {onlyfiles_test[k]}, PSNR: {PSNR:.2f} dB, ssim: {ssim_value :.4f} dB")

                mean_PSNR = np.mean(list_PSNR)
                mean_PSNR_init = np.mean(list_PSNR_init)
                mean_ssim = np.mean(list_SSIM)

                print(f"Mean PSNR for {dataset}: {mean_PSNR:.2f} dB")
                print(f"Mean PSNR_init for {dataset}: {mean_PSNR_init:.2f} dB")
                print(f"Mean ssim for {dataset}: {mean_ssim:.4f} dB")
                file_to_print.write("FINAL" + " " + str(mean_PSNR) + "\n")
                file_to_print.write("FINAL" + " " + str(mean_PSNR_init) + "\n")
                file_to_print.write("FINAL" + " " + str(mean_ssim) + "\n")
                file_to_print.flush()
            end = time.time()
            occur = end - start
            print(f'{dataset}时间：{occur}')
            file_to_print.close()

    break

print("Finished Testing")
