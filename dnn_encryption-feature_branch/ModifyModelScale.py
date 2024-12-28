import torch
import torch.nn as nn
import copy
import random
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import psutil
class Net(nn.Module):
    def __init__(self, input_features, output_features):
        super(Net, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.regression = nn.Linear(
            in_features=self.input_features, out_features=self.output_features
        )

    def forward(self, x):
        x = self.regression(x)
        return x

def print_memory_usage():
    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def ModifyModelScale(net, Scale, split_num=4):
    device = next(net.parameters()).device
    print(f"Device: {device}")

    TTB = net.regression.weight
    BiaB = net.regression.bias

    print(f"Original weight shape: {TTB.shape}")
    r, c = TTB.shape

    # 克隆原始偏置并创建新的假偏置空矩阵
    BSave = BiaB.clone()
    BSaveB = torch.zeros(r * 3, device=device)
    MaxBias = BSave.abs().max()

    for i in range(r * 3):
        random_indices = torch.randint(0, r, (7,)) # 随机选择7个原始偏置
        avg_value = BSave[random_indices].mean()  # 计算平均值
        Flag = 1 if avg_value >= 0 else -1  # 处理符号
        new_value = (MaxBias - abs(avg_value)) * Scale * Flag
        BSaveB[i] = new_value

    WSave = TTB.clone()
    WSaveB = torch.zeros(r * 3, c, device=device) 
    WeightMax = WSave.abs().max()

    for i in range(r * 3):
        random_indices = torch.randint(0, r, (7,)) #改变这个量可以改变合成的数量
        avg_values = WSave[random_indices].mean(dim=0)
        
        for j in range(c):
            Flag = 1 if avg_values[j] >= 0 else -1
            new_value = (WeightMax - abs(avg_values[j])) * Scale * Flag
            WSaveB[i, j] = new_value

    OutL = r * 4 
    NewWeight = torch.cat([WSave, WSaveB], dim=0)
    NewBias = torch.cat([BSave, BSaveB], dim=0)

    order_mapping = []
    print("======Updating weights and biases======")
    with torch.no_grad():
        for i in range(OutL):
            threshold = torch.rand(1).item()  # 生成0到1之间的随机数
            if i < r and random.random() < threshold:  # 对于前 r 个位置，有机会放真实特征
                NewWeight[i] = WSave[i]
                NewBias[i] = BSave[i]
                order_mapping.append(('original', i))
            else:  # 其余位置放假特征
                fake_index = random.randint(0, r * 3 - 1)
                NewWeight[i] = WSaveB[fake_index]
                NewBias[i] = BSaveB[fake_index]
                order_mapping.append(('fake', i))

    print("First 10 items in order_mapping:", order_mapping[:10])
    print("Starting to create split models...")

    split_size = OutL // split_num
    # 创建分割模型
    for i in range(split_num):
        print(f"Generating model {i+1}/{split_num}")
        new_model = copy.deepcopy(net)
        start_index = i * split_size
        end_index = OutL if i == split_num - 1 else (i + 1) * split_size

        # 创建新的线性层，使用分割后的权重和偏置
        new_regression_part = nn.Linear(c, end_index - start_index, bias=True).to(device)
        new_regression_part.weight.data = NewWeight[start_index:end_index, :].clone()
        new_regression_part.bias.data = NewBias[start_index:end_index].clone()
        # 替换模型的回归层
        new_model.regression = new_regression_part
        new_model.order_mapping = order_mapping
        # 定义新的前向传播函数，去除softmax
        def new_forward(self, x):
            return self.regression(x)

        new_model.forward = new_forward.__get__(new_model, type(new_model))

        print(f"Model {i+1} parameters: {sum(p.numel() for p in new_model.parameters())}")

        yield new_model  # 使用生成器逐个返回模型

def DecodeOutput(models, outputs):
    device = outputs[0].device
    total_outputs = torch.cat(outputs, dim=1)
    
    original_feature_count = 40
    decoded_output = torch.zeros(total_outputs.shape[0], original_feature_count, device=device)
    
    order_mapping = models[0].order_mapping
    original_indices = [i for i, (type, _) in enumerate(order_mapping) if type == 'original']
    
    for i, (type, index) in enumerate(order_mapping):
        if type == 'original' and i < total_outputs.shape[1]:
            decoded_output[:, index] = total_outputs[:, i]
    
    return decoded_output

def evaluate_models(models, dataloader):
    device = next(models[0].parameters()).device
    for model in models:
        model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = [model(images.reshape(images.shape[0], -1)) for model in models]
            decoded_output = DecodeOutput(models, outputs)
            _, predicted = torch.max(decoded_output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def apply_defense(model_path, Scale=30, split_num=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_model = Net(112 * 92, 40)
    original_model.load_state_dict(torch.load(model_path, map_location=device))
    original_model.to(device)

    print("Applying defense mechanism...")
    
    defended_models = []
    for i, model in enumerate(ModifyModelScale(original_model, Scale, split_num)):
        print(f"Processing model {i+1}/{split_num}")
        defended_models.append(model)
        
        save_dir = os.path.dirname(model_path)
        torch.save({
            'model_state_dict': model.state_dict(),
            'order_mapping': model.order_mapping
        }, os.path.join(save_dir, f"defended_model_part_{i}.pkl"))
        
        print(f"Model {i+1} saved")
        print_memory_usage()

    print("Defense mechanism applied and all models saved")
    return defended_models

if __name__ == "__main__":
    model_path = r"C:\Users\ADMIN\lpl\MIA\logs\Model_2024Jun24_03-29-17_ADMIN-20240528F_main AT and T face\mynet_50.pkl"
    dataset_dir = r"C:\Users\ADMIN\lpl\MIA\at&t face database"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    dataset = ImageFolder(dataset_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    try:
        defended_models = apply_defense(model_path)
        print(f"Number of defended models: {len(defended_models)}")
        accuracy = evaluate_models(defended_models, dataloader)
        print(f"Accuracy after defense: {accuracy:.4f}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

    print("Main program completed")