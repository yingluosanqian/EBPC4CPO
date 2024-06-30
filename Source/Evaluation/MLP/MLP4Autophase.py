import sys
sys.path.append('./../../../')
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
        weights = F.softmax(self.attention_weights, dim=0)
        x = x * weights
        return x


class MLPWithAttention(nn.Module):
    def __init__(self, input_dim=124, hidden_dim=256, output_dim=50):
        super(MLPWithAttention, self).__init__()
        self.attention = AttentionModule(input_dim)
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
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
    has_nan = torch.isnan(X).any()
    return X, Y


def load_val(filepath1, filepath2):
    X = np.loadtxt(filepath1, delimiter=",")
    Y = np.loadtxt(filepath2, delimiter=",")
    X = torch.FloatTensor(X)
    Y = torch.FloatTensor(Y)
    has_nan = torch.isnan(X).any()
    return X, Y


def mlp_training(model, X_train, Y_train, num_epochs=100, batch_size=100, lr=5e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(num_epochs):
        loss_ls = []
        for i, (inputs, labels) in enumerate(dataloader):
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            optimizer.zero_grad()
            loss.backward()

            loss_ls.append(loss.item())
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {sum(loss_ls) / len(loss_ls)}')


def mlp_testing(model, X_test, Y_test):
    def get_max_index(ls):
        return torch.argmax(ls)

    def is_kth_largest(ls, index, k):
        unique_elements = sorted(set(ls), reverse=True)
        rank = unique_elements.index(ls[index]) + 1
        return 1 if rank <= k else 0

    model.eval()

    with torch.no_grad():
        predictions = model(X_test)

    for top in [1, 3, 5]:
        v = sum([(is_kth_largest(Y, get_max_index(prediction), top)) for prediction, Y in tqdm(zip(predictions, Y_test), desc='Model Testing')])
        print(f'top{top}: {v / len(Y_test)}')


def mlp_predict(model, X, k):
    model.eval()
    with torch.no_grad():
        predictions = model(X)
        print(predictions)
    res = []
    for prediction in predictions:
        _, indices = torch.topk(prediction, k)
        res.append(indices)
    return res


def model_saving(saved_model, model_path):
    torch.save(saved_model.state_dict(), model_path)
    print('Model saved!')


def model_loading(model, model_path):
    model.load_state_dict(torch.load(model_path))
    print('Model loaded!')
