import pandas as pd
from glob import glob
import re
import numpy as np
import os

target_path = "data/MNIST_result/re_ex/"

root_paths = sorted(glob( target_path + "experiment/**"))
root_paths = [s for s in root_paths if ('train' in s) or ('val' in s)]

print(root_paths)

for root_path in root_paths:

    all_train = np.zeros(48000)
    all_val = np.zeros(12000)
    all_test = np.zeros(10000)

    csv_paths = sorted(glob(root_path + "/**.csv"),
                        key=lambda s: int(re.findall(r'\d+', s)[len(re.findall(r'\d+', s))-1]))

    train_paths = [s for s in csv_paths if 'train_result' in s]
    val_paths = [s for s in csv_paths if 'val_result' in s]
    test_paths = [s for s in csv_paths if 'test_result' in s]
    print(train_paths)
    print(val_paths)
    print(test_paths)

    root_name = os.path.basename(root_path)

    for (train_path, val_path, test_path) in zip(train_paths, val_paths, test_paths):
    # for (train_path, val_path) in zip(train_paths, val_paths):

        train = pd.read_csv(train_path).values[:, 1].astype(np.int)
        val = pd.read_csv(val_path).values[:, 1].astype(np.int)
        test = pd.read_csv(test_path).values[:, 1].astype(np.int)

        all_train += train
        all_val += val
        all_test += test

    # csvに保存
    train_history = pd.DataFrame({"Tの数": all_train}, index = [i for i in range(48000)])
    train_history.to_csv( target_path + "result/" + root_name + "/train.csv")
    val_history = pd.DataFrame({"Tの数": all_val}, index = [i for i in range(12000)])
    val_history.to_csv( target_path + "result/" + root_name + "/val.csv")
    test_history = pd.DataFrame({"Tの数": all_test}, index = [i for i in range(10000)])
    test_history.to_csv( target_path + "result/" + root_name + "/test.csv")

