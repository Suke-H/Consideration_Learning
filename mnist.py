# パッケージのimport
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

# 学習結果の保存用
history = {
    'train_acc': [],
    'val_acc': [],
    'test_acc': [],
}


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
        self.dropout1 = torch.nn.Dropout(0.5)
        self.dropout2 = torch.nn.Dropout(0.5)
 
    def forward(self, x):
        # テンソルのリサイズ: (N, 1, 28, 28) --> (N, 784)
        x = x.view(-1, 28 * 28) 

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
 
        return F.log_softmax(x, dim=1)

def train_model(net, dataloaders_dict, criterion, optimizer, limit_phase, limit_acc, num_epochs):
    """
    モデルを訓練

    Arguments:
        net : モデル
        dataloaders_dict : DataLoaderを辞書型にしたもの
        criterion : 損失関数
        optimizer : 最適アルゴリズム
        num_epochs : エポック数

    """
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
            for i, (inputs, labels) in tqdm(enumerate(dataloaders_dict[phase])):

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
                    # numpyの一次配列に変換
                    batch_tf = batch_tf.cpu().numpy()
                    batch_tf.reshape(batch_tf.shape[0])
                    # train_tf / val_tfに代入
                    if phase == "train":
                        train_tf[i*_batch_size : (i+1)*_batch_size] = batch_tf
                    else:
                        val_tf[i*_batch_size : (i+1)*_batch_size] = batch_tf
                    
            # epochごとのlossと正解率を表示
            #print("{}-len: {}".format(phase, len(dataloaders_dict[phase].dataset)))
            print("collect: {}".format(epoch_corrects))
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double(
            ) / len(dataloaders_dict[phase].dataset)

            # historyに保存
            history[phase + '_acc'].append(epoch_acc)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 下限accを越したら学習停止
            if epoch_acc >= limit_acc and phase == limit_phase:
                print("{}_acc : {} >= {}\nstop at {} epoch!".format(phase, epoch_acc, limit_acc, epoch + 1))
                stop_flag = True

        # 停止フラグが立ったら終了
        if stop_flag:
            break

    # train_tf/val_tfをcsvに出力
    train_df = pd.DataFrame({"TF": train_tf}, index = [i for i in range(len(dataloaders_dict["train"].dataset))])
    train_df.to_csv("data/train_result.csv")
    val_df = pd.DataFrame({"TF": val_tf}, index = [i for i in range(len(dataloaders_dict["val"].dataset))])
    val_df.to_csv("data/val_result.csv")

    # 何epochで終わったか出力
    return epoch + 1

def test_model(net, dataloaders_dict):
    net.eval()  # または net.train(False) でも良い
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in dataloaders_dict['test']:
            data = data.view(-1, 28 * 28)
            # print(data.shape)
            output = net(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            # print(pred.shape)
            # a = input()
            
            # targetとpredが合っていたらカウントアップ
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= 10000

    print('Test loss (avg): {}, Accuracy: {}'.format(test_loss,
                                                        correct / 10000))

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

print("train:{}, val:{}, test:{}".format(len(train_dataset), len(val_dataset), len(test_dataset)))

# # 前処理しないデータセットと比較                                    
# sample_set = torchvision.datasets.MNIST(root='./data', 
#                                         train=True,
#                                         download=True,
#                                         transform=None
#                                         )
                              
# sample2_set = torchvision.datasets.MNIST(root='./data', 
#                                         train=True,
#                                         download=True,
#                                         transform=transforms.ToTensor()
#                                         )

# img2 =  sample2_set[59999][0]
# print(type(img2), img2.shape)
# a = input()
# img2 =  img2.cpu().numpy().reshape(28, 28)

# img1 = sample_set[59999][0]
# img2 =  trainset[59999][0].cpu().numpy().reshape(28, 28)
# plt.imshow(img1, cmap = 'gray')
# plt.show()
# plt.imshow(img2, cmap = 'gray')
# plt.show()

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
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# 学習
num_epochs = 5
limit_phase = "val"
limit_acc = 0.75
attempt_num = train_model(model, dataloaders_dict, criterion, optimizer, limit_phase, limit_acc, num_epochs)

plt.figure()
plt.plot(range(attempt_num), history['train_acc'], label='train_acc')
plt.plot(range(attempt_num), history['val_acc'], label='val_acc')
plt.xlabel('epoch')
plt.xlabel('acc')
plt.legend()
plt.savefig('data/acc_test.png')

# 検証
test_model(model, dataloaders_dict)


