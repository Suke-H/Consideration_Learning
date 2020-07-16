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

def check_tf(net, x, t, path):
    # xを入力する準備
    # _input = np.array(x[:, np.newaxis, :, np.newaxis], dtype="float32")
    _input = np.array(x, dtype="float32")
    _input = torch.from_numpy(_input)
    
    t = torch.from_numpy(t)

    # xをnetに入力し、labelを出力
    _output = net(_input)
    _, pred = torch.max(_output, 1)  # ラベルを予測
    
    # tと比較し、T/F(1/0)にする 
    tf = pred.eq(t.view_as(pred))

    # csvに出力
    df = pd.DataFrame({"TF": tf.cpu().numpy()}, index = [i for i in range(len(t))])
    df.to_csv(path)


def fix_train_model(net, dataloaders_dict, data_dict, criterion, optimizer, limit_phase, limit_acc, num_epochs, root_path):
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

            # データローダーからミニバッチを取り出すループ
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

    # train, valを入力し、結果のTFをcsvに出力
    check_tf(net, data_dict["train_x"], data_dict["train_y"], path=root_path+"train_result_"+str(file_no)+".csv")
    check_tf(net, data_dict["val_x"], data_dict["val_y"], path=root_path+"val_result_"+str(file_no)+".csv")

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

def fix_test_model(net, dataloaders_dict, data_dict, input_dim, root_path):
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


    # train, valを入力し、結果のTFをcsvに出力
    check_tf(net, data_dict["test_x"], data_dict["test_y"], path=root_path+"test_result_"+str(file_no)+".csv")

    test_loss /= test_size
    test_acc = correct / test_size

     # test_accを記録
    with open(root_path+"test_acc.csv", 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([file_no, test_acc])

    print('Test loss (avg): {}, Accuracy: {}'.format(test_loss, test_acc))

def compile_data(target_path):
    """
    各試行の正解数を足す
    target_pathはその下にexperience(各試行のcsvファイルが格納)とresultフォルダ(出力用)があることが前提
    
    """
    # target_path = "data/artifact/animation/"

    root_paths = sorted(glob( target_path + "experiment/**"))
    root_paths = [s for s in root_paths if ('train' in s) or ('val' in s)]

    print(root_paths)

    for root_path in root_paths:

        all_train = np.zeros(10000)
        all_val = np.zeros(1000)
        all_test = np.zeros(1000)

        csv_paths = sorted(glob(root_path + "/**.csv"),
                            key=lambda s: int(re.findall(r'\d+', s)[len(re.findall(r'\d+', s))-1]))

        train_paths = [s for s in csv_paths if 'train_result' in s]
        val_paths = [s for s in csv_paths if 'val_result' in s]
        # test_paths = [s for s in csv_paths if 'test_result' in s]

        root_name = os.path.basename(root_path)

        for (train_path, val_path) in zip(train_paths, val_paths):
        # for (train_path, val_path, test_path) in zip(train_paths, val_paths, test_paths):

            train = pd.read_csv(train_path).values[:, 1]
            val = pd.read_csv(val_path).values[:, 1]
            # test = pd.read_csv(test_path).values[:, 1]

            all_train += train
            all_val += val
            # all_test += test

        # csvに保存
        train_history = pd.DataFrame({"Tの数": all_train}, index = [i for i in range(10000)])
        train_history.to_csv( target_path + "result/" + root_name + "/train.csv")
        val_history = pd.DataFrame({"Tの数": all_val}, index = [i for i in range(1000)])
        val_history.to_csv( target_path + "result/" + root_name + "/val.csv")
        # test_history = pd.DataFrame({"Tの数": all_test}, index = [i for i in range(1000)])
        # test_history.to_csv( target_path + "result/" + root_name + "/test.csv")
