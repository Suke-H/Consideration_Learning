import pandas as pd
import numpy as np
import csv

with open("data/test.csv", 'a', newline="") as f:
    writer = csv.writer(f)
    writer.writerow([1, 5])

# train_df =  pd.read_csv('data/val75/test_result.csv').values
# val_df = pd.read_csv('data/val_result.csv').values

# train_df = train_df[:, 1]
# val_df = val_df[:, 1]

# train_corrects = np.count_nonzero(train_df == 1)
# val_corrects = np.count_nonzero(val_df == 1)

# train_acc = train_corrects / 48000
# val_acc = val_corrects / 12000

# print("train: {}, val: {}".format(train_acc, val_acc))

test_df =  pd.read_csv('data/val75/test_result_11.csv').values

test_df = test_df[:, 1]

test_corrects = np.count_nonzero(test_df == 1)

test_acc = test_corrects / 10000

print(test_acc)
