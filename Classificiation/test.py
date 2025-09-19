import os
import torch
import torch.distributed as dist
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

class Tester:
    def __init__(self, model, dataloader_val, device):
        self.model = model
        self.dataloader_val = dataloader_val
        self.device = device

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        all_labels = []
        all_predicted_out = []

        pbar_test = tqdm(self.dataloader_val, desc="Testing")
        for patches_t, labels in pbar_test:
            labels = labels.unsqueeze(1)  # labels 的形状变为 [batch_size, 1]
            labels = labels.float()  # labels 的形状变为 [batch_size, 1]
            patches, labels = patches_t.to(self.device), labels.to(self.device)

            # 模型预测
            predicted = self.model(patches)
            predicted = torch.sigmoid(predicted)

            # 根据 0.5 的阈值将概率转换为类别标签
            predicted_out = (predicted >= 0.5).float()
            total += labels.size(0)
            correct += (predicted_out == labels).sum().item()

            # 保存所有标签和预测结果
            all_labels.extend(labels.cpu().numpy())
            all_predicted_out.extend(predicted_out.cpu().numpy())

        # 计算精确率、召回率和 F1 值
        precision = precision_score(all_labels, all_predicted_out)
        recall = recall_score(all_labels, all_predicted_out)
        f1 = f1_score(all_labels, all_predicted_out)
        accuracy = correct / total

        return accuracy, precision, recall, f1


# Example usage in the main function
if __name__ == "__main__":
    # Initialize Tester class
    tester = Tester()


