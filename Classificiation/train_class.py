
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from dataset import get_train_transform, get_test_transform
from scipy import linalg
from sklearn.metrics import precision_score, recall_score, f1_score
from dataset import TumorDataset
from utils import setup_logger
from test import *
from Classificiation import DS_DKSVD_cla


def init_distributed_mode(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)

def main(rank, world_size):
    init_distributed_mode(rank, world_size)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(f"cuda:{rank}")
    # 调用外部文件中定义的转换函数
    train_transform = get_train_transform()
    test_transform = get_test_transform()

    # 创建训练集和测试集的Dataset对象
    train_dataset = TumorDataset(image_folder='dataset/train', transform=train_transform)
    # Dataloader of the training set:
    batch_size = 4
    sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=world_size,
                                                  rank=dist.get_rank(), shuffle=True,
                                                  drop_last=True)
    dataloader_train = DataLoader(
        train_dataset, batch_size=int(batch_size), shuffle=False, num_workers=world_size,sampler=sampler,drop_last=True
    )

    val_dataset = TumorDataset(image_folder='dataset/val',  transform=test_transform)
    dataloader_val  = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    test_dataset = TumorDataset(image_folder='dataset/test', transform=test_transform)
    dataloader_test = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Initialization:
    patch_size = 16
    m = 20
    Dict_init = DS_DKSVD_cla.init_dct(patch_size, m)
    Dict_init = Dict_init.to(device)

    c_init = (linalg.norm(Dict_init.cpu(), ord=2)) ** 2
    c_init = torch.FloatTensor((c_init,))
    c_init = c_init.to(device)

    D_in, H_1, H_2,  D_out_lam, D_out_lam_pd,T, min_v, max_v, indice = patch_size ** 2, 512, 768, 112, 400, 7, -1, 1, 112
    model = DS_DKSVD_cla.DenoisingNet_MLP_3(
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
        indice,
        device,
    )


    if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = DDP(model, device_ids=[rank])#, find_unused_parameters=True

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    # criterion = torch.nn.MSELoss(reduction="mean")
    criterion_clf =torch.nn.BCELoss()


    # 初始化日志记录器
    log_folder = 'LOG'
    logger = setup_logger(log_folder, 'train_log.log')

    # Training loop
    epochs = 200
    running_loss = 0.0
    alpha = 0
    train_losses, test_losses = [], []

    # 从 `start_epoch` 开始训练
    for epoch in range(epochs):  # 确保从 start_epoch 开始训练
        model_filename = f"train_weight"
        if not os.path.exists(model_filename):
            os.makedirs(model_filename, exist_ok=True)
        pbar = tqdm(enumerate(dataloader_train), total=len(dataloader_train), desc=f"Epoch {epoch + 1}/{epochs}")
        for i, (image_t, labels) in pbar:
            # 使用 unsqueeze(1) 将 labels 的形状修改为 [batch_size, 1]
            labels = labels.unsqueeze(1)  # labels 的形状变为 [batch_size, 1]
            labels = labels.float()  # labels 的形状变为 [batch_size, 1]
            image_t, labels = image_t.to(device),labels.to(device)

            optimizer.zero_grad()
            predicted = model(image_t)
            predicted = torch.sigmoid(predicted)

            # loss_image = criterion(outputs.to(device),image_t)
            loss = criterion_clf(predicted, labels)
            # loss = loss_clf+ alpha * loss_image
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.6f}"})

        # 汇总所有GPU的损失
        loss_tensor = torch.tensor(running_loss).to(rank)
        dist.reduce(loss_tensor, dst=0, op=dist.ReduceOp.SUM)


        if rank == 0:
            average_train_losses = loss_tensor /(len(dataloader_train)*world_size)
            with torch.no_grad():
                test_loss = 0
                model.eval()
                correct = 0
                total = 0
                all_labels = []
                all_predicted_out = []

                pbar_test = tqdm(dataloader_val, desc="Testing")
                for patches_t, labels in pbar_test:
                    labels = labels.unsqueeze(1)  # labels 的形状变为 [batch_size, 1]
                    labels = labels.float()  # labels 的形状变为 [batch_size, 1]
                    patches,  labels = (
                        patches_t.to(device),
                        labels.to(device)
                    )
                    predicted = model(patches)
                    predicted = torch.sigmoid(predicted)
                    # 根据 0.5 的阈值将概率转换为类别标签
                    predicted_out = (predicted >= 0.5).float()
                    total += labels.size(0)
                    correct += (predicted_out == labels).sum().item()
                    # 计算损失
                    # loss_image = criterion(outputs, patches)
                    loss= criterion_clf(predicted, labels)
                    # loss = loss_clf + alpha * loss_image
                    test_loss += loss.item()
                test_loss = test_loss / len(dataloader_val)

        if rank == 0:
            tester = Tester(model, dataloader_test, device)
            accuracy, precision, recall, f1 = tester.evaluate()

            logger.info(
                f"Rank {rank}: Epoch [{epoch + 1}/{epochs}], "f"Loss:{average_train_losses:.8f},"
                f"test:{test_loss:.8f}, Accuracy:{accuracy:.4f}, "f"Precision:{precision:.4f}, Recall:{recall:.4f}, F1 Score:{f1:.4f}"
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

        if rank == 0:
            epoch_filename = os.path.join(model_filename, f"epoch_{epoch + 1}.pth")
            torch.save(state, epoch_filename)
            # logger.info(f"Model for epoch {epoch + 1} saved to {epoch_filename}")
    # Save model if needed
    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    # world_size = 1
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
