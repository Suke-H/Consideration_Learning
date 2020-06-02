from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import os
from tqdm import tqdm
import pathlib

# フォルダ取得
root_paths = sorted(glob("data/artifact/experiment/**/**"))
root_paths = [s for s in root_paths if 'epoch' in s]

for root_path in root_paths:

    parent = pathlib.Path(root_path).parent.name
    print(parent)

    acc = pd.read_csv(root_path).values[:, 2]

    acc_mean = np.mean(acc)
    acc_var = np.var(acc) * 10000

    print("mean: {}, var: {}".format(acc_mean, acc_var))

