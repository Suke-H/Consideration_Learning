from PIL import Image
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from tqdm import tqdm

def display_artifact(x, nums, fig_path):

    print("path:{}".format(fig_path))

    if "entire" in fig_path:
        index0 = np.where(nums >= 30)
        index1 = np.where((29 >= nums) & (nums >= 20))
        index2 = np.where((19 >= nums) & (nums >= 10))
        index3 = np.where(9 >= nums)

        label0 = "30~40"
        label1 = "20~29"
        label2 = "10~19"
        label3 = "0~9"

    elif "div" in fig_path:
        index0 = np.where(nums == 1)
        index1 = []
        index2 = np.where(nums == 0)
        index3 = []

        label0 = "1"
        label1 = "-"
        label2 = "0"
        label3 = "-"

    else:
        index0 = np.where(nums >= 4)
        index1 = np.where(nums == 3)
        index2 = np.where(nums == 2)
        index3 = np.where(1 >= nums)

        label0 = "4, 5"
        label1 = "3"
        label2 = "2"
        label3 = "0, 1"

    x0, x1, x2, x3 = x[index0], x[index1], x[index2], x[index3]

    # xy軸
    plt.plot([-1, 1], [0, 0], marker=".",color="black")
    plt.plot([0, 0], [-1, 1], marker=".",color="black")

    plt.plot(x0[:, 0],x0[:, 1],marker="o",linestyle="None",color="red", label=label0)
    plt.plot(x1[:, 0],x1[:, 1],marker="o",linestyle="None",color="orange", label=label1)
    plt.plot(x2[:, 0],x2[:, 1],marker="o",linestyle="None",color="blue", label=label2)
    plt.plot(x3[:, 0],x3[:, 1],marker="o",linestyle="None",color="purple", label=label3)

    # 凡例の表示
    plt.legend()
    
    plt.show()
    # plt.savefig(fig_path)
    plt.close()


# 人工データ読み込み
train_x = np.load("data/artifact/dataset/train_x.npy")
# train_y = np.load("data/artifact/dataset/train_y.npy")
val_x = np.load("data/artifact/dataset/val_x.npy")
# val_y = np.load("data/artifact/dataset/val_y.npy")
test_x = np.load("data/artifact/dataset/test_x.npy")
# test_y = np.load("data/artifact/dataset/test_y.npy")

# フォルダ取得
root_path = "data/artifact/re_ex/experiment/val80"

# csvを読み込み
train_df =  pd.read_csv(root_path + '/train_result_1.csv')
# "TF"で昇順ソート
train_df = train_df.sort_values('TF')
# numpyに変換
_, train_nums = train_df.values.T

val_df =  pd.read_csv(root_path + '/val_result_1.csv')
val_df = val_df.sort_values('TF')
val_indices, val_nums = val_df.values.T

test_df =  pd.read_csv(root_path + '/test_result_1.csv')
test_df = test_df.sort_values('TF')
test_indices, test_nums = test_df.values.T

# 図に出力
display_artifact(train_x, train_nums, root_path + "/train_last.png")
display_artifact(val_x, val_nums, root_path + "/val_last.png")
display_artifact(test_x, test_nums, root_path + "/test_last.png")
