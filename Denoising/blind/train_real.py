import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import transforms
from scipy import linalg
import numpy as np
import logging
from skimage.metrics import structural_similarity
import blind_dataset
import real_blind_dataset
from model import DS_DKSVD


def init_distributed_mode(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)

def main(rank, world_size):
    init_distributed_mode(rank, world_size)


    # Your existing code
    # List of the test image names BSD68:
    file_test_raw = open("FMD_dataset/TwoPhoton_MICE/real_test_raw.txt", "r")
    test_raw = []
    for e in  file_test_raw:
        test_raw.append(e[:-1])


    file_test_gt = open("FMD_dataset/TwoPhoton_MICE/real_test_gt.txt", "r")
    test_gt = []
    for e in  file_test_gt:
        test_gt.append(e[:-1])

    # List of the train image names:
    file_train_raw = open("FMD_dataset/TwoPhoton_MICE/real_train_raw.txt", "r")
    train_raw = []
    for e in file_train_raw:
        train_raw.append(e[:-1])

    file_train_gt = open("FMD_dataset/TwoPhoton_MICE/real_train_gt.txt", "r")
    train_gt = []
    for e in file_train_gt:
        train_gt.append(e[:-1])



    set12_test_file = open("d_Test_txt/test_set12.txt", "r")
    set12_file = []
    for e in set12_test_file:
        set12_file.append(e[:-1])

    # Rescaling in [-1, 1]:
    mean = 255 / 2
    std = 255 / 2
    data_transform = transforms.Compose(
        [blind_dataset.Normalize(mean=mean, std=std), blind_dataset.ToTensor()]
    )
    # Noise level:
    sigma = 25
    # Sub Image Size:
    sub_image_size = 128


    # Training Dataset:
    my_Data_train = real_blind_dataset.SubImagesDataset(
        noisy_dir="FMD_dataset/TwoPhoton_MICE/train/raw",
        clean_dir="FMD_dataset/TwoPhoton_MICE/train/gt",
        noisy_names= train_raw,
        clean_names= train_gt,
        sub_image_size=sub_image_size,
        stride=1,
        transform=data_transform,
    )
    # Test Dataset:
    my_Data_test =real_blind_dataset.FullImagesDataset(
        noisy_dir="FMD_dataset/TwoPhoton_MICE/test/raw",
        clean_dir="FMD_dataset/TwoPhoton_MICE/test/gt",
        noisy_names=test_raw,
        clean_names=test_gt, transform=data_transform
    )

    # Dataloader of the test set:
    num_images_test = 5
    indices_test = np.random.randint(0, 50, num_images_test).tolist()
    my_Data_test_sub = torch.utils.data.Subset(my_Data_test, indices_test)
    dataloader_test = DataLoader(
        my_Data_test_sub, batch_size=1, shuffle=False, num_workers=0
    )

    # Dataloader of the training set:
    batch_size = 6
    sampler = torch.utils.data.DistributedSampler(my_Data_train, num_replicas=world_size,
                                                  rank=dist.get_rank(), shuffle=True,
                                                  drop_last=True)

    dataloader_train = DataLoader(
        my_Data_train, batch_size=int(batch_size), shuffle=False, num_workers=world_size,sampler=sampler
    )

    dataloader_test_zong = real_blind_dataset.FullImagesDataset(
        noisy_dir="FMD_dataset/TwoPhoton_MICE/test/raw",
        clean_dir="FMD_dataset/TwoPhoton_MICE/test/gt",
        noisy_names=test_raw,
        clean_names=test_gt, transform=data_transform
    )

    dataloader_test_zong = DataLoader( dataloader_test_zong, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device("cuda", rank)
    file_to_print = open("results_training.csv", "w")
    file_to_print.write(str(device) + "\n")
    file_to_print.flush()

    
    # Initialization:
    patch_size = 16
    m = 20
    Dict_init = blind_dataset.init_dct(patch_size, m)
    # checkpoint = torch.load('model_230.pth')
    # Dict_init = checkpoint['model_state_dict']['module.Dict']
    Dict_init = Dict_init.to(device)

    c_init = (linalg.norm(Dict_init.cpu(), ord=2)) ** 2
    c_init = torch.FloatTensor((c_init,))
    c_init = c_init.to(device)
    #


    # D_in, H_1, H_2, H_3, D_out_lam, T, min_v, max_v = patch_size ** 2, 512, 256, 128, 1, 7, -1, 1
    D_in, H_1, H_2,  D_out_lam, D_out_lam_pd,T, min_v, max_v, indice = patch_size ** 2, 512, 768, 112, 400, 7, -1, 1,112
    model = DS_DKSVD.DenoisingNet_MLP_3(
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
        c_init,
        # w_init1 ,
        # w_init2,
        # w_init3 ,
        indice,
        device,
    )


    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = DDP(model, device_ids=[rank])#, find_unused_parameters=True

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    criterion = torch.nn.MSELoss(reduction="mean")

    # 指定日志文件保存的文件夹路径
    log_folder = 'log'
    os.makedirs(log_folder, exist_ok=True)  # 如果文件夹不存在，则创建

    # 创建第一个日志文件的记录器
    logger1 = logging.getLogger('logger1')
    logger1.setLevel(logging.INFO)
    file_handler1 = logging.FileHandler(os.path.join(log_folder, 'TwoPhoton_MICE_train.log'))
    stream_handler1 = logging.StreamHandler()  # 添加一个 StreamHandler 输出到控制台
    formatter1 = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler1.setFormatter(formatter1)
    stream_handler1.setFormatter(formatter1)  # 格式化控制台输出
    logger1.addHandler(file_handler1)
    logger1.addHandler(stream_handler1)  # 将 StreamHandler 添加到 logger1

    # 创建第二个日志文件的记录器
    logger2 = logging.getLogger('logger2')
    logger2.setLevel(logging.INFO)
    file_handler2 = logging.FileHandler(os.path.join(log_folder, 'TwoPhoton_MICE_psnr.log'))
    stream_handler2 = logging.StreamHandler()  # 添加一个 StreamHandler 输出到控制台
    formatter2 = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler2.setFormatter(formatter2)
    stream_handler2.setFormatter(formatter2)  # 格式化控制台输出
    logger2.addHandler(file_handler2)
    logger2.addHandler(stream_handler2)  # 将 StreamHandler 添加到 logger2

    def InNormalization(image):
        mean = 255 / 2
        std = 255 / 2
        return image * std + mean



    # Training loop
    epochs = 3
    running_loss = 0.0
    print_every = 100
    # alpha = 0.00001
    train_losses, test_losses = [], []

    # total_steps = len(dataloader_train)*epochs
    # #----------------------------yuxiantuihuadiaoduqi-----------------------------------
    # optimizer = optim.SGD(model.parameters(), lr=2*1e-4,momentum=0.5)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    for epoch in range(epochs):
        model_filename = f"TwoPhoton_MICE_train_Modle"
        if not os.path.exists(model_filename):
            os.makedirs(model_filename,exist_ok=True)
        for i, (sub_images, sub_images_noise) in enumerate(dataloader_train):

            sub_images, sub_images_noise = sub_images.to(device), sub_images_noise.to(device)
            optimizer.zero_grad()
            outputs = model(sub_images_noise)
            loss = criterion(outputs.to(device), sub_images)
            # loss = loss1 + alpha * orthogonal_loss
            loss.backward()
            optimizer.step()


            running_loss += loss.item()


            if i % print_every == print_every-1:
                # 汇总所有GPU的损失
                loss_tensor = torch.tensor(running_loss).to(rank)
                dist.reduce(loss_tensor, dst=0, op=dist.ReduceOp.SUM)

                if rank == 0 :
                    average_train_losses = loss_tensor / (print_every * world_size)
                    with torch.no_grad():
                        test_loss = 0
                        model.eval()
                        for patches_t, patches_noise_t in dataloader_test:
                            patches, patches_noise = (
                                patches_t.to(device),
                                patches_noise_t.to(device),
                            )
                            outputs = model(patches_noise)
                            loss = criterion(outputs, patches)
                            # loss = loss1 + alpha * orthogonal_loss
                            test_loss += loss.item()
                        test_loss = test_loss / len(dataloader_test)

                    logger1.info(
                        f"Rank {rank}: Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(dataloader_train)}], Loss:{average_train_losses:.8f}, 交叉验证:{test_loss:.8f},"
                    )

                running_loss = 0.0

                # 保存模型状态
                state = ({
                    'iter': i,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                })
                torch.save(state, os.path.join(model_filename, 'latest.pth'))

                if (i + 1) % (100*print_every) == 0:
                    model_save_count = int(i / 10000)
                    torch.save(state, os.path.join(model_filename, f"model_{model_save_count}.pth"))

            if rank == 0 and (i + 1) % (100*print_every) == 0:
                with torch.no_grad():
                    list_PSNR = []
                    list_SSIM = []
                    model.eval()
                    for k, (image_true, image_noise) in enumerate(dataloader_test_zong, 0):
                        image_true_t = image_true[0, 0, :, :].to(device)
                        image_noise_t = image_noise.to(device)
                        image_restored_t = model(image_noise_t)
                        image_restored_t = image_restored_t[0, 0, :, :]
                        PSNR = 10 * torch.log10(4 / torch.mean((image_true_t - image_restored_t) ** 2)).cpu()
                        list_PSNR.append(PSNR)

                        # Calculate SSIM
                        img1 = InNormalization(image_true_t.cpu()).numpy().astype('uint8')
                        real_denoise = InNormalization(image_restored_t.cpu()).numpy().astype('uint8')
                        ssim_value = structural_similarity(img1, real_denoise, data_range=255)
                        list_SSIM.append(ssim_value)

                    mean_PSNR = np.mean(list_PSNR)
                    mean_SSIM = np.mean(list_SSIM)
                    # print(f"Image {k + 1}/{len(dataloader_test)} set12_mean_PSNR: {mean_PSNR:.3f} dB")
                    logger2.info(
                        f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(dataloader_train)}],Test_mean_PSNR: {mean_PSNR:.2f} dB,Test_mean_SSIM: {mean_SSIM:.4f} ")


    # Save model if needed
    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    # world_size = 4
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
