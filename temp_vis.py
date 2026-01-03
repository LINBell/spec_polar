import os
import numpy as np
from matplotlib import pyplot as plt

filter_path = 'mask/99_100_100_full_matrix.npy' #更换为自己矩阵
data = np.load(filter_path)  # c h w
print(data.shape)  # (99, 100, 100)