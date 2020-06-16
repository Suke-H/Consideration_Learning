# パッケージのimport
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import csv
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

from test_artifact import SimpleNet, ArtifactDataset, ArtifactTransform

def grid_plot(x, output, fig_path, data=None):

    index0 = np.where(output == 0)
    index1 = np.where(output == 1)
    index2 = np.where(output == 2)
    index3 = np.where(output == 3)

    label0 = "0"
    label1 = "1"
    label2 = "2"
    label3 = "3"

    x0, x1, x2, x3 = x[index0], x[index1], x[index2], x[index3]

    plt.plot(x0[:, 0],x0[:, 1],marker=".",linestyle="None",color="red", label=label0)
    plt.plot(x1[:, 0],x1[:, 1],marker=".",linestyle="None",color="blue", label=label1)
    plt.plot(x2[:, 0],x2[:, 1],marker=".",linestyle="None",color="yellow", label=label2)
    plt.plot(x3[:, 0],x3[:, 1],marker=".",linestyle="None",color="purple", label=label3)

    if data is not None:
        plt.plot(data[:, 0],data[:, 1],marker="o",linestyle="None",color="white", label="validation_data")

    # xy軸
    plt.plot([-1, 1], [0, 0], marker=".",color="black")
    plt.plot([0, 0], [-1, 1], marker=".",color="black")

    # 凡例の表示
    plt.legend()
    
    # plt.show()
    plt.savefig(fig_path)
    plt.close()

def visualization(net, dataloaders_dict, criterion, optimizer, limit_phase, limit_acc, num_epochs, root_path, val_x, 
                    grid_num=100, vis_epoch=5):

    """
    モデルを訓練

    Arguments:
        net : モデル
        dataloaders_dict : DataLoaderを辞書型にしたもの
        criterion : 損失関数
        optimizer : 最適アルゴリズム
        num_epochs : エポック数

    """

    # 学習結果の保存用
    history = {
        'train_acc': [],
        'val_acc': [],
        'test_acc': []
    }

    # 何回目の試行か
    file_no = len(glob(root_path+"**.png"))

    ani_path = root_path + "/" + str(file_no) + "/"
    os.mkdir(ani_path)

    # 停止フラグ
    stop_flag = False

    # データセットに正解1/不正解0でラベル付け
    train_tf = np.zeros(len(dataloaders_dict["train"].dataset))
    val_tf = np.zeros(len(dataloaders_dict["val"].dataset))

    # 格子点を入力する準備
    grid_elem = np.linspace(-1, 1, grid_num)

    xx = np.array([[x for x in grid_elem] for _ in range(grid_num)])
    xx = xx.reshape(grid_num**2)
    yy = np.array([[y for _ in range(grid_num)] for y in grid_elem])
    yy = yy.reshape(grid_num**2)

    xy = np.stack([xx, yy])
    xy = xy.T
    grid_input = np.array(xy[:, np.newaxis, :, np.newaxis], dtype="float32")
    grid_input = torch.from_numpy(grid_input)

    # val_xを入力する準備
    val_input = np.array(val_x[:, np.newaxis, :, np.newaxis], dtype="float32")
    val_input = torch.from_numpy(val_input)

    # epochのループ
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        # epochごとに学習とバリデーション
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
            else:
                net.eval()   # モデルを検証モードに

            if (epoch % vis_epoch == 0) and (phase=="train"):

                # 格子点を入力にする
                grid_output = net(grid_input)
                _, grid_pred = torch.max(grid_output, 1)  # ラベルを予測

                # 可視化
                grid_plot(xy, grid_pred.cpu().numpy(), ani_path+str(epoch)+".png", data=val_x)

            epoch_loss = 0.0  # epochの損失和
            epoch_corrects = 0  # epochの正解数

            # データローダーからミニバッチを取り出すループ
            # for i, (inputs, labels) in enumerate(tqdm(dataloaders_dict[phase])):
            for i, (inputs, labels) in enumerate(dataloaders_dict[phase]):
                _batch_size = inputs.size(0)

                # optimizerを初期化
                optimizer.zero_grad()

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = net(inputs)
                    loss = criterion(outputs, labels)  # 損失を計算
                    _, preds = torch.max(outputs, 1)  # ラベルを予測
                    
                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # イタレーション結果の計算
                    # lossの合計を更新
                    epoch_loss += loss.item() * _batch_size 
                    # 正解数の合計を更新
                    epoch_corrects += torch.sum(preds == labels.data)

                    # ミニバッチごとのデータセットに正解1/不正解0でラベル付け
                    batch_tf = preds.eq(labels)
                    # numpyに変換
                    batch_tf = batch_tf.cpu().numpy()

                    # train_tf / val_tfに代入
                    if phase == "train":
                        train_tf[i*_batch_size : (i+1)*_batch_size] = batch_tf
                    else:
                        val_tf[i*_batch_size : (i+1)*_batch_size] = batch_tf
                    
            # epochごとのlossと正解率を表示
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double(
            ) / len(dataloaders_dict[phase].dataset)

            # historyに保存
            history[phase + '_acc'].append(epoch_acc.item())
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 上限accを越したら学習停止
            if epoch_acc >= limit_acc and phase == limit_phase:
                print("{}_acc : {} >= {}\nstop at {} epoch!".format(phase, epoch_acc, limit_acc, epoch + 1))
                stop_flag = True

        # 停止フラグが立ったら終了
        if stop_flag:
            break

    end_epoch = epoch + 1

    if stop_flag == False:
        print("learning_epoch has reached the upper_limit_epoch {} before {}_acc reached the upper_limit_acc {} ...".format(end_epoch, phase, limit_acc))

    # 格子点を入力にする
    grid_output = net(grid_input)
    _, grid_pred = torch.max(grid_output, 1)  # ラベルを予測

    # 可視化
    grid_plot(xy, grid_pred.cpu().numpy(), ani_path+"end"+str(epoch)+".png", data=val_x)

    # val_xを入力にする
    val_output = net(val_input)
    _, val_pred = torch.max(val_output, 1)  # ラベルを予測
    # 可視化
    grid_plot(val_x, val_pred.cpu().numpy(), ani_path+"end_val.png")

    # train_tf/val_tfをcsvに出力
    train_df = pd.DataFrame({"TF": train_tf}, index = [i for i in range(len(dataloaders_dict["train"].dataset))])
    train_df.to_csv(root_path+"train_result_"+str(file_no)+".csv")
    val_df = pd.DataFrame({"TF": val_tf}, index = [i for i in range(len(dataloaders_dict["val"].dataset))])
    val_df.to_csv(root_path+"val_result_"+str(file_no)+".csv")

    # 学習記録(epoch-acc)をプロット
    plt.figure()
    runs = [i for i in range(1, end_epoch+1)]
    plt.plot(runs, history['train_acc'], label='train_acc')
    plt.plot(runs, history['val_acc'], label='val_acc')
    plt.plot(runs, [limit_acc for i in range(end_epoch)], label='limit_acc')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.xticks(runs)
    plt.legend()
    plt.savefig(root_path+"acc_"+str(file_no)+".png")

    # csvにも保存
    train_history = pd.DataFrame({"acc": history['train_acc']}, index = [i for i in range(end_epoch)])
    train_history.to_csv(root_path+"train_history_"+str(file_no)+".csv")
    val_history = pd.DataFrame({"acc": history['val_acc']}, index = [i for i in range(end_epoch)])
    val_history.to_csv(root_path+"val_history_"+str(file_no)+".csv")

if __name__ == "__main__":

    # 設定するパラメータ等
    limit_phase = "val"
    limit_acc = 0.80
    lr=10**(-4)
    momentum = 0.5
    num_epochs = 100
    root_path = "data/artifact/animation/experiment/val80_2/"

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
    visualization(model, dataloaders_dict, criterion, optimizer, limit_phase, limit_acc, num_epochs, root_path, val_x, vis_epoch=5)

