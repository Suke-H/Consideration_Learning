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

# 学習結果の保存用
history = {
    'train_acc': [],
    'val_acc': [],
    'test_acc': []
}

# 設定するパラメータ等
limit_phase = "train"
limit_acc = 0.75
lr=10**(-2)
num_epochs = 100
root_path = "data/train75/"

end_epoch = 0
file_no = len(glob(root_path+"*.png"))

def Random(a, b):
    """ aからbまでの一様乱数を返す """
    return (b - a) * np.random.rand() + a

def make_artificial_data(n, phase):
    """
    -1 <= x, y <= 1での一様乱数により作られた2次元のデータを

    label 0: 第1象限(右上)
    label 1: 第2象限(左上)
    label 2: 第3象限(左下)
    label 3: 第4象限(右下)

    の4クラスにしたデータ

    train: 各ラベルの数を同じにしたデータ生成
    val, test: ランダムにデータ生成

    """

    # trainデータ生成
    if phase == "train":
        # 正解ラベル
        labels = np.array([int(i/int(n/4)) for i in range(n)])
        # データ
        dataset = np.zeros((n, 2))

        for i, label in enumerate(labels):

            # label 0
            if label == 0:
                dataset[i] = [Random(0, 1), Random(0, 1)]

            # label 1
            elif label == 1:
                dataset[i] = [Random(-1, 0), Random(0, 1)]

            # label 2
            elif label == 2:
                dataset[i] = [Random(-1, 0), Random(-1, 0)]

            # label 3
            else:
                dataset[i] = [Random(0, 1), Random(-1, 0)]

        # シャッフル
        perm = np.random.permutation(n)
        dataset, labels = dataset[perm], labels[perm]

    # val, testデータ生成
    else:
        # データ
        dataset = np.array([[Random(-1, 1), Random(-1, 1)] for i in range(n)])

        # 正解ラベル
        labels = np.zeros(n)

        for i, data in enumerate(dataset):
            x, y = data

            # label 0
            if x >= 0 and y >= 0:
                labels[i] = 0

            # label 1
            elif x < 0 and y >= 0:
                labels[i] = 1

            # label 2
            elif x < 0 and y < 0:
                labels[i] = 2

            # label 3
            else:
                labels[i] = 3

    return np.array(dataset, dtype="float32"), np.array(labels, dtype="int")

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

        return data, int(label)

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
        data = data + np.full_like(data, 1)
        # ToTensor用に次元を増やす
        data = np.array([[data[0]], [data[1]]])

        return self.data_transform(data)

# モデル定義
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = torch.nn.Linear(2, 100)
        self.fc2 = torch.nn.Linear(100, 4)
 
    def forward(self, x):
        # テンソルのリサイズ: (N, 1, 2, 1) --> (N, 2)
        x = x.view(-1, 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)

if __name__ == "__main__":

    # 人工データ作成
    train_x, train_y = make_artificial_data(10000, phase="train")
    val_x, val_y = make_artificial_data(1000, phase="val")
    test_x, test_y = make_artificial_data(1000, phase="test")

    # 入力画像の前処理用の関数
    transform = ArtifactTransform()
    data_transformed = transform

    # torchのDatasetクラスにする
    train_dataset = ArtifactDataset(train_x, train_y, transform=data_transformed)
    val_dataset = ArtifactDataset(val_x, val_y, transform=data_transformed)
    test_dataset = ArtifactDataset(val_x, val_y, transform=data_transformed)

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

    # モデル定義
    model = SimpleNet()
    # 損失関数
    criterion = torch.nn.CrossEntropyLoss()
    # 最適アルゴリズム
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)

    # 学習
    train_model(model, dataloaders_dict, criterion, optimizer, limit_phase, limit_acc, num_epochs)

    # 検証
    test_model(model, dataloaders_dict, 2)

