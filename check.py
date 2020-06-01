import pandas as pd
from glob import glob
import re
import numpy as np

root_paths = sorted(glob("data/artifact/experiment/**"))
root_paths = [s for s in root_paths if ('train' in s) or ('val' in s)]
print(root_paths)

all_train = np.zeros(10000)
all_val = np.zeros(1000)
all_test = np.zeros(1000)

for root_path in root_paths:
    csv_paths = sorted(glob(root_path + "/**.csv"),
                        key=lambda s: int(re.findall(r'\d+', s)[len(re.findall(r'\d+', s))-1]))

    train_paths = [s for s in csv_paths if 'train_result' in s]
    val_paths = [s for s in csv_paths if 'val_result' in s]
    test_paths = [s for s in csv_paths if 'test_result' in s]

    for (train_path, val_path, test_path) in zip(train_paths, val_paths, test_paths):

        train = pd.read_csv(train_path).values[:, 1]
        val = pd.read_csv(val_path).values[:, 1]
        test = pd.read_csv(test_path).values[:, 1]

        all_train += train
        all_val += val
        all_test += test

# csvに保存
train_history = pd.DataFrame({"Tの数": all_train}, index = [i for i in range(10000)])
train_history.to_csv("data/artifact/result/entire/train.csv")
val_history = pd.DataFrame({"Tの数": all_val}, index = [i for i in range(1000)])
val_history.to_csv("data/artifact/result/entire/val.csv")
test_history = pd.DataFrame({"Tの数": all_test}, index = [i for i in range(1000)])
test_history.to_csv("data/artifact/result/entire/test.csv")

