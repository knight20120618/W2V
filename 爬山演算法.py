# %%
import numpy as np
import random

def Hill(x):
    old = np.zeros(x)
    new = np.zeros(x)
    z = 0 # 計算次數
    while new.sum() != x:
        y = random.randint(0, len(new)-1)
        new[y] = 1
        if new.sum() > old.sum():
            old = new
            z += 1
            print('{} => {}'.format(old, old.sum()))
        else:
            z += 1
            print('{}=> {}'.format(old, old.sum()))
    print('共 {} 次'.format(z))

# 輸入數值
Hill(10)


