# パッケージのimport
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

from mnist import ImageTransform, train_model, test_model
from modules import fix_train_model, fix_test_model

class ArtifactDataset(data.Dataset):
    """
    -1 <= x, y <= 1での一様乱数により作られた2次元のデータを

    label 0: 第1象限(右上)
    label 1: 第2象限(左上)
    label 2: 第3象限(左下)
    label 3: 第4象限(右下)

    の4クラスにしたデータ

    """

    def __init__(self, dataset, labels, transform=None):
        self.dataset = dataset
        self.labels = labels
        self.transform = transform  # 前処理クラスのインスタンス

    def __len__(self):
        '''データの数を返す'''
        return len(self.labels)

    def __getitem__(self, index):
        '''
        前処理をした画像のTensor形式のデータとラベルを取得
        '''
        # データ
        data = self.dataset[index]
        label = self.labels[index]

        data = self.transform(data)

        return data, label

class ArtifactTransform():
    """
    画像の前処理クラス。
    torchテンソル化のみ行う。

    """

    def __init__(self):
        self.data_transform = transforms.Compose(
                            [transforms.ToTensor()])

    def __call__(self, data):
        # 全体に[1, 1]を足して正の値のみにする
        # data = data + np.full_like(data, 1)
        # ToTensor用に次元を増やす
        data = np.array([[data[0]], [data[1]]])

        return self.data_transform(data)

# モデル定義
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = torch.nn.Linear(2, 64)
        self.fc2 = torch.nn.Linear(64, 4)
 
    def forward(self, x):
        # テンソルのリサイズ: (N, 1, 2, 1) --> (N, 2)
        x = x.view(-1, 2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":

    # 設定するパラメータ等
    limit_phase = "val"
    limit_acc = 0.50
    lr=10**(-4)
    momentum = 0.5
    num_epochs = 100
    root_path = "data/artifact/re_ex/val50/"

    # 人工データ読み込み
    train_x = np.load("data/artifact/dataset/train_x.npy")
    train_y = np.load("data/artifact/dataset/train_y.npy")
    val_x = np.load("data/artifact/dataset/val_x.npy")
    val_y = np.load("data/artifact/dataset/val_y.npy")
    test_x = np.load("data/artifact/dataset/test_x.npy")
    test_y = np.load("data/artifact/dataset/test_y.npy")

    # 入力画像の前処理用の関数
    transform = ArtifactTransform()
    data_transformed = transform

    # torchのDatasetクラスにする
    train_dataset = ArtifactDataset(train_x, train_y, transform=data_transformed)
    val_dataset = ArtifactDataset(val_x, val_y, transform=data_transformed)
    test_dataset = ArtifactDataset(test_x, test_y, transform=data_transformed)

    print("train:{}, val:{}, test:{}".format(len(train_dataset), len(val_dataset), len(test_dataset)))

    # DataLoaderの作成
    train_loader = torch.utils.data.DataLoader(train_dataset,
                    batch_size=100, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, 
                    batch_size=100, shuffle=False, num_workers=2)                                         
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                    batch_size=100, shuffle=False, num_workers=2)

    # 辞書型変数にまとめる
    dataloaders_dict = {"train": train_loader, "val": val_loader, "test": test_loader}

    # 生データも辞書型に
    data_dict = {"train_x": train_x, "val_x": val_x, "test_x": test_x, 
                    "train_y": train_y, "val_y": val_y, "test_y": test_y}

    # モデル定義
    model = SimpleNet()
    # 損失関数
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.MSELoss()
    # 最適アルゴリズム
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    # optimizer = optim.RMSprop(model.parameters())
    # optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-09)

    # 学習
    fix_train_model(model, dataloaders_dict, data_dict, criterion, optimizer, limit_phase, limit_acc, num_epochs, root_path)

    # 検証
    fix_test_model(model, dataloaders_dict, data_dict, 2, root_path)

