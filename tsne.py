import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd

import torch
import torchvision

from mnist import ImageTransform, Subset

def tSNE(x, t):

    print(x.shape, t.shape)

    X_reduced = TSNE(n_components=2, random_state=0).fit_transform(x)
    print(X_reduced.shape)

    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t)
    plt.colorbar()

    plt.show()

if __name__ == "__main__":

    # 入力画像の前処理用の関数
    transform = ImageTransform()
    img_transformed = transform

    # データセット読み込み + 前処理
    trainval_dataset = torchvision.datasets.MNIST(root='./data', 
                        train=True, download=True, transform=img_transformed)

    # 元々trainデータのものをtrain/valに分割
    n_samples = len(trainval_dataset) # n_samples is 60000
    train_size = int(n_samples * 0.8) # train_size is 48000

    subset1_indices = list(range(0,train_size)) # [0,1,.....47999]
    subset2_indices = list(range(train_size,n_samples)) # [48000,48001,.....59999]

    train_dataset = Subset(trainval_dataset, subset1_indices)
    val_dataset   = Subset(trainval_dataset, subset2_indices)

    # データセット読み込み + 前処理
    test_dataset = torchvision.datasets.MNIST(root='./data', 
                    train=False, download=True, transform=img_transformed)

    # train_x = np.array([train_dataset[i][0].cpu().numpy() for i in range(48000)]).reshape(48000, 28*28)
    # val_x = np.array([val_dataset[i][0].cpu().numpy() for i in range(12000)]).reshape(12000, 28*28)
    # test_x = np.array([test_dataset[i][0].cpu().numpy() for i in range(10000)]).reshape(10000, 28*28)

    # train_t = pd.read_csv("data/MNIST_result/re_ex/experiment/val80/train_result_5.csv").values[:, 1].astype(np.int)
    # val_t = pd.read_csv("data/MNIST_result/re_ex/experiment/val80/val_result_5.csv").values[:, 1].astype(np.int)
    # test_t = pd.read_csv("data/MNIST_result/re_ex/experiment/val80/test_result_5.csv").values[:, 1].astype(np.int)

    train_x = np.array([train_dataset[i][0].cpu().numpy() for i in range(1000)]).reshape(1000, 28*28)
    val_x = np.array([val_dataset[i][0].cpu().numpy() for i in range(1000)]).reshape(1000, 28*28)
    test_x = np.array([test_dataset[i][0].cpu().numpy() for i in range(1000)]).reshape(1000, 28*28)

    # train_t = np.array([train_dataset[i][1] for i in range(1000)])
    # val_t = np.array([val_dataset[i][1] for i in range(1000)])
    # test_t = np.array([test_dataset[i][1] for i in range(1000)])

    train_t = pd.read_csv("data/MNIST_result/re_ex/result/val80/train.csv").values[:1000, 1].astype(np.int)
    val_t = pd.read_csv("data/MNIST_result/re_ex/result/val80/val.csv").values[:1000, 1].astype(np.int)
    test_t = pd.read_csv("data/MNIST_result/re_ex/result/val80/test.csv").values[:1000, 1].astype(np.int)

    tSNE(train_x, train_t)
    tSNE(val_x, val_t)
    tSNE(test_x, test_t)