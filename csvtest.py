import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

end_epoch = 10
history = {
    "a": [0.4+0.01*i for i in range(end_epoch)], 
    "b": [0.3+0.03*i for i in range(end_epoch)]
}

# 学習記録(epoch-acc)をプロット
plt.figure()
plt.plot(range(end_epoch), history['a'], label='train_acc')
plt.plot(range(end_epoch), history['b'], label='val_acc')
plt.plot(range(end_epoch), [0.5 for i in range(end_epoch)], label='limit_acc')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend()
plt.savefig("data/testacc.png")
