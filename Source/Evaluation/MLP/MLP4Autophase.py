import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from pathlib import Path

from Source.Util.util import get_root_path


# 定义网络结构
class MLP(nn.Module):
    def __init__(self, input_size=124, hidden=128, output_size=50):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_size),
        )

    def forward(self, x):
        return self.layers(x)


class AttentionModule(nn.Module):
    def __init__(self, input_dim):
        super(AttentionModule, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(input_dim))

    def forward(self, x):
        # 计算注意力权重，使用softmax使得所有权重和为1
        weights = F.softmax(self.attention_weights, dim=0)
        # 应用注意力权重到输入特征上
        x = x * weights
        return x


class MLPWithAttention(nn.Module):
    def __init__(self, input_dim=124, hidden_dim=256, output_dim=50):
        super(MLPWithAttention, self).__init__()
        self.attention = AttentionModule(input_dim)
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 应用注意力模块
        x = self.attention(x)
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x


def load_train(filepath1, filepath2, T=0.0015):
    X = np.loadtxt(filepath1, delimiter=",")
    Y = np.loadtxt(filepath2, delimiter=",")
    X = torch.FloatTensor(X)
    Y = torch.FloatTensor(Y)
    Y = F.softmax(Y / T, dim=1)
    # 检查是否存在NaN值
    has_nan = torch.isnan(X).any()
    print(f'张量中是否存在NaN值: {has_nan}')
    return X, Y


def load_val(filepath1, filepath2):
    X = np.loadtxt(filepath1, delimiter=",")
    Y = np.loadtxt(filepath2, delimiter=",")
    X = torch.FloatTensor(X)
    Y = torch.FloatTensor(Y)
    # 检查是否存在NaN值
    has_nan = torch.isnan(X).any()
    print(f'张量中是否存在NaN值: {has_nan}')
    return X, Y


def mlp_training(model, X_train, Y_train, num_epochs=100, batch_size=100, lr=5e-4):
    # 实例化模型、损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 适用于二元分类的交叉熵损失，假定你的任务是多标签分类
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 创建TensorDataset和DataLoader
    dataset = TensorDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(num_epochs):
        loss_ls = []
        for i, (inputs, labels) in enumerate(dataloader):
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)

            # 后向传播和优化
            optimizer.zero_grad()
            loss.backward()

            loss_ls.append(loss.item())
            optimizer.step()

        # 打印统计信息
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {sum(loss_ls) / len(loss_ls)}')


def mlp_testing(model, X_test, Y_test):
    def get_max_index(ls):
        return torch.argmax(ls)

    def is_kth_largest(ls, index, k):
        unique_elements = sorted(set(ls), reverse=True)  # 找出唯一元素并降序排序
        rank = unique_elements.index(ls[index]) + 1  # +1是因为排名从1开始
        return 1 if rank <= k else 0

    # 设置模型为评估模式
    model.eval()

    # 使用模型进行预测
    with torch.no_grad():  # 不追踪梯度，因为我们不在训练模式
        predictions = model(X_test)
    # print(f'predictions shape: {predictions.shape}')

    for top in [1, 3, 5]:
        v = sum([(is_kth_largest(Y, get_max_index(prediction), top)) for prediction, Y in tqdm(zip(predictions, Y_test), desc='Model Testing')])
        print(f'top{top}: {v / len(Y_test)}')


def mlp_predict(model, X, k):
    # 设置模型为评估模式
    model.eval()

    # print(f'k: {k}')

    # 使用模型进行预测
    with torch.no_grad():  # 不追踪梯度，因为我们不在训练模式
        predictions = model(X)
        print(predictions)
    res = []
    for prediction in predictions:
        _, indices = torch.topk(prediction, k)
        res.append(indices)
    return res


def model_saving(saved_model, model_path):
    # 导出模型参数到文件
    torch.save(saved_model.state_dict(), model_path)
    print('Model saved!')


def model_loading(model, model_path):
    # 导入模型参数
    model.load_state_dict(torch.load(model_path))
    print('Model loaded!')


if __name__ == '__main__':

    # 假设我们有一些数据
    # X_train 和 y_train 需要是torch.Tensor类型，分别代表输入和目标输出
    # X_train 的形状应该是 [n_samples, 124]
    # y_train 的形状应该是 [n_samples, 50]
    # 以下是训练循环的伪代码
    np.random.seed(998244353)
    torch.manual_seed(998244353)

    # X_train, Y_train = load_train('train_X.csv', 'train_Y.csv')
    # X_val, Y_val = load_test('val_X.csv', 'val_Y.csv')
    # model = MLP(input_size=56)
    # mlp_training(model, X_train, Y_train, batch_size=500, num_epochs=150, lr=5e-3)
    # mlp_training(model, X_train, Y_train, batch_size=500, num_epochs=150, lr=9e-4)
    # model_saving(model, 'autophase_mlp')
    # mlp_testing(model, X_val, Y_val)

    # X_train, Y_train = load_train('data_124/train_final_X.csv', 'data_124/train_final_Y.csv')
    # X_val, Y_val = load_test('data_124/val_final_X.csv', 'data_124/val_final_Y.csv')
    # print(f'X_train shape: {X_train.shape}')
    # print(f'Y_train shape: {Y_train.shape}')
    # model = MLPWithAttention()
    # mlp_training(model, X_train, Y_train, batch_size=500, num_epochs=500, lr=5e-3)
    # mlp_testing(model, X_val, Y_val)
    # model_saving(model, 'ylsq_attn_mlp')

    # X_train, Y_train = load_train('data_124/train_coreset_X.csv', 'data_124/train_coreset_Y.csv')
    # X_val, Y_val = load_test('data_124/val_coreset_X.csv', 'data_124/val_coreset_Y.csv')
    # print(f'X_train shape: {X_train.shape}')
    # print(f'Y_train shape: {Y_train.shape}')
    # model = MLPWithAttention()
    # mlp_training(model, X_train, Y_train, batch_size=500, num_epochs=500, lr=5e-3)
    # mlp_testing(model, X_val, Y_val)
    # model_saving(model, 'coreset_attn_mlp')

    X_train, Y_train = load_train('data_124/train_final_X.csv', 'data_124/train_final_Y.csv')
    X_val, Y_val = load_val('data_124/val_final_X.csv', 'data_124/val_final_Y.csv')
    print(f'X_train shape: {X_train.shape}')
    print(f'Y_train shape: {Y_train.shape}')
    model = MLP()
    mlp_training(model, X_train, Y_train, batch_size=500, num_epochs=150, lr=5e-3)
    mlp_testing(model, X_val, Y_val)
    model_saving(model, 'ylsq_mlp')

    X_train, Y_train = load_train('data_124/train_coreset_X.csv', 'data_124/train_coreset_Y.csv')
    X_val, Y_val = load_val('data_124/val_coreset_X.csv', 'data_124/val_coreset_Y.csv')
    print(f'X_train shape: {X_train.shape}')
    print(f'Y_train shape: {Y_train.shape}')
    model = MLP()
    mlp_training(model, X_train, Y_train, batch_size=500, num_epochs=150, lr=5e-3)
    mlp_testing(model, X_val, Y_val)
    model_saving(model, 'coreset_mlp')
