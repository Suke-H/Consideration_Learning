import numpy as np
import matplotlib.pyplot as plt

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