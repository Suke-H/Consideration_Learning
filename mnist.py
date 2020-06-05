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

class ImageTransform():
    """
    画像の前処理クラス。
    torchテンソル化と標準化を行う。

    """

    def __init__(self):
        self.data_transform = transforms.Compose(
                            [transforms.ToTensor(),
                            transforms.Normalize((0.5, ), (0.5, ))])

    def __call__(self, img):
        return self.data_transform(img)

class Subset(torch.utils.data.Dataset):
    """
    インデックスを入力にデータセットの部分集合を取り出す

    Arguments:
        dataset : 入力データセット
        indices : 取り出すデータセットのインデックス
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

# モデル定義
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 1000)
        self.fc2 = torch.nn.Linear(1000, 10)
 
    def forward(self, x):
        # テンソルのリサイズ: (N, 1, 28, 28) -> (N, 784)
        x = x.view(-1, 28 * 28) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)

def train_model(net, dataloaders_dict, criterion, optimizer, limit_phase, limit_acc, num_epochs, root_path):
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

    file_no = len(glob(root_path+"*.png"))

    # 停止フラグ
    stop_flag = False

    # データセットに正解1/不正解0でラベル付け
    train_tf = np.zeros(len(dataloaders_dict["train"].dataset))
    val_tf = np.zeros(len(dataloaders_dict["val"].dataset))

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
            # net.eval()

            epoch_loss = 0.0  # epochの損失和
            epoch_corrects = 0  # epochの正解数

            # # 未学習時の検証性能を確かめるため、epoch=0の訓練は省略
            # if (epoch == 0) and (phase == 'train'):
            #     continue

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

def test_model(net, dataloaders_dict, input_dim, root_path):
    file_no = len(glob(root_path+"*.png"))

    net.eval()  # または net.train(False) でも良い
    test_loss = 0
    correct = 0

    test_size = len(dataloaders_dict["test"].dataset)

    # データセットに正解1/不正解0でラベル付け
    test_tf = np.zeros(test_size)

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(dataloaders_dict["test"])):
            _batch_size = inputs.size(0)

            inputs = inputs.view(-1, input_dim)
            output = net(inputs)
            test_loss += F.nll_loss(output, labels, reduction='sum').item()
            preds = output.argmax(dim=1, keepdim=True)
            
            # targetとpredが合っていたらカウントアップ
            correct += preds.eq(labels.view_as(preds)).sum().item()

            # ミニバッチごとのデータセットに正解1/不正解0でラベル付け
            batch_tf = preds.eq(labels.view_as(preds))
            # numpyの一次配列に変換
            batch_tf = batch_tf.cpu().numpy()
            batch_tf = batch_tf.flatten()
            
            # test_tfに代入
            test_tf[i*_batch_size : (i+1)*_batch_size] = batch_tf

    # train_tf/val_tfをcsvに出力
    test_df = pd.DataFrame({"TF": test_tf}, index = [i for i in range(test_size)])
    test_df.to_csv(root_path+"test_result_"+str(file_no)+".csv")

    test_loss /= test_size
    test_acc = correct / test_size

     # test_accを記録
    with open(root_path+"test_acc.csv", 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([file_no, test_acc])

    print('Test loss (avg): {}, Accuracy: {}'.format(test_loss, test_acc))

if __name__ == "__main__":

    limit_phase = "train"
    limit_acc = 0.75
    lr = 10**(-2)
    num_epochs = 10
    root_path = "data/MNIST_result/experiment/train75/"

    # 入力画像の前処理用の関数

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
    model = Net()
    # 損失関数
    criterion = torch.nn.CrossEntropyLoss()
    # 最適アルゴリズム
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)

    # 学習
    train_model(model, dataloaders_dict, criterion, optimizer, limit_phase, limit_acc, num_epochs, root_path)

    # 検証
    test_model(model, dataloaders_dict, 28*28, root_path)

