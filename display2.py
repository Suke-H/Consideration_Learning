from PIL import Image
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from glob import glob
import os
from tqdm import tqdm

from mnist import Subset

def display30(dataset, indices, nums, fig_path):

    #表示領域を設定（行，列）
    fig, ax = plt.subplots(5, 6)  

    #図を配置
    for i, (idx, num) in enumerate(zip(indices, nums)):
        idx, num = int(idx), int(num)
        plt.subplot(5,6,i+1)
        title = str(idx) + "(" + str(num) + ")"
        plt.title(title, fontsize=10)    #タイトルを付ける
        plt.tick_params(color='white')      #メモリを消す
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        plt.imshow(train_dataset[idx][0], cmap="gray")   #図を入れ込む

    #図が重ならないようにする
    plt.tight_layout()
    #図を表示
    #plt.show()
    #保存
    plt.savefig(fig_path)
    # pdfファイルの初期化
    #pp = PdfPages(fig_path)

    # figureをセーブする
    #pp.savefig()

    plt.close()

# データセット読み込み + 前処理
trainval_dataset = torchvision.datasets.MNIST(root='./data', 
                    train=True, download=True, transform=None)

# 元々trainデータのものをtrain/valに分割
n_samples = len(trainval_dataset) # n_samples is 60000
train_size = int(n_samples * 0.8) # train_size is 48000

subset1_indices = list(range(0,train_size)) # [0,1,.....47999]
subset2_indices = list(range(train_size,n_samples)) # [48000,48001,.....59999]

train_dataset = Subset(trainval_dataset, subset1_indices)
val_dataset   = Subset(trainval_dataset, subset2_indices)

# データセット読み込み + 前処理
test_dataset = torchvision.datasets.MNIST(root='./data', 
                train=False, download=True, transform=None)

# フォルダ取得
root_paths = sorted(glob("data/MNIST_result/re_ex/result/**"))
root_paths = [s for s in root_paths if ('train' in s) or ('val' in s)]

print(root_paths)

for root_path in tqdm(root_paths):

    # csvを読み込み
    train_df =  pd.read_csv(root_path + '/train.csv')
    # "Tの数"で昇順ソート
    train_df = train_df.sort_values('Tの数')
    # numpyに変換
    train_indices, train_nums = train_df.values.T

    val_df =  pd.read_csv(root_path + '/val.csv')
    val_df = val_df.sort_values('Tの数')
    val_indices, val_nums = val_df.values.T

    test_df =  pd.read_csv(root_path + '/test.csv')
    test_df = test_df.sort_values('Tの数')
    test_indices, test_nums = test_df.values.T

    # worst30を図に出力
    display30(train_dataset, train_indices[:30], train_nums[:30], root_path + "/train_false.png")
    display30(val_dataset, val_indices[:30], val_nums[:30], root_path + "/val_false.png")
    display30(test_dataset, test_indices[:30], test_nums[:30], root_path + "/test_false.png")

    # 降順ソート
    train_indices, train_nums = train_indices[::-1], train_nums[::-1]
    val_indices, val_nums = val_indices[::-1], val_nums[::-1]
    test_indices, test_nums = test_indices[::-1], test_nums[::-1]

    # best30を図に出力
    display30(train_dataset, train_indices[:30], train_nums[:30], root_path + "/train_true.png")
    display30(val_dataset, val_indices[:30], val_nums[:30], root_path + "/val_true.png")
    display30(test_dataset, test_indices[:30], test_nums[:30], root_path + "/test_true.png")
