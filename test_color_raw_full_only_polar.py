import random

from scipy.sparse import data
from torch import nn as nn, true_divide
from tqdm import tqdm
from matplotlib import pyplot as plt, colors
from architecture import *
# from simulation.train_code.temp_test import stack_21
from utils import *
import torch
import scipy.io as scio
import time
import os
import numpy as np
from torch.autograd import Variable
import datetime
from option import opt
import torch.nn.functional as F





def cal_dop(img):
    """
    input [c, s, h, w]
    """
    S0, S1, S2, S3 = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :], img[:, 3, :, :]
    temp = np.sqrt(S1**2 + S2**2 + S3**2)
    dop = np.zeros_like(temp)
    valid_mask = (S0 > 0)
    dop[valid_mask] = temp[valid_mask] / S0[valid_mask]
    return np.clip(dop, 0, 1)


def stack_21(img):
    band1 = img[:,:,2]
    band2 = img[:,:,10]
    band3 = img[:,:,18]
    data = np.stack([band1, band2, band3], axis=-1)
    return data


def cal_restric(data):
    s0 = data[..., 0]  # S0分量
    s1 = data[..., 1]  # S1分量
    s2 = data[..., 2]  # S2分量
    s3 = data[..., 3]  # S3分量
    # 计算不等式两边
    left_side = s0 ** 2
    right_side = s1 ** 2 + s2 ** 2 + s3 ** 2

    # 检查条件是否满足
    condition_met = left_side >= right_side

    # 统计满足条件的像素和光谱点数量
    total_points = np.prod(data.shape[:-1])  # 总点数(512*612*21)
    valid_points = np.sum(condition_met)  # 满足条件的点数
    invalid_points = total_points - valid_points  # 不满足条件的点数

    print(f"总检查点数: {total_points}")
    print(f"满足 S0² ≥ S1²+S2²+S3² 的点数: {valid_points} ({valid_points / total_points:.2%})")
    print(f"不满足条件的点数: {invalid_points} ({invalid_points / total_points:.2%})")


def np_psnr(img, ref):  #  c s h w 4 4 200 200
    img = (img * 256).round().clip(0, 255)
    ref = (ref * 256).round().clip(0, 255)
    nC= img.shape[0]
    psnr = 0
    for i in range(nC):
        mse = np.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        psnr += 10 * np.log10((255*255)/mse)
    return psnr / (nC)


batch_size = 1


model = MST(dim=4, stage=2, num_blocks=[4, 7, 5], need_noise=True, num_sp=5, need_mask_atten=False, raw_flag=True).cuda()


model_path = 'ablation_test/only_polar/model/model_epoch_231.pth'
model_path = 'ablation_test/polar_gen_data_mst/model/model_epoch_279.pth'

model.load_state_dict(torch.load(model_path))


filter_path = 'mask/99_hand_bd_matrix.npy'
filter_path = 'mask/99_100_100_full_matrix.npy'
# filter_path = 'mask/99_100_100_full_matrix_4T.npy' #更换为自己矩阵
# filter_path = 'mask/99_100_100_full_matrix_10T.npy' #更换为自己矩阵
filter_matrix = torch.from_numpy(np.load(filter_path)).unsqueeze(0).repeat(opt.batch_size, 1, 1, 1, 1).cuda().float()
filter_matrix_test = torch.from_numpy(np.load(filter_path)).unsqueeze(0).repeat(batch_size, 1, 1, 5, 5).cuda().float()
filter_matrix = filter_matrix[:, :, :, :opt.crop_size, :opt.crop_size]
# filter_matrix = filter_matrix[:, :, :, :290, :380]
# filter_matrix_test = filter_matrix_test[:, :, :, :300, :300]

def stokes_filter_to_hvadlr_torch(filter_stokes):
    """
    使用PyTorch直接实现转换，避免CPU-GPU数据传输
    """
    bs, c, s, h, w = filter_stokes.shape
    
    # 提取各个斯托克斯分量
    S0 = filter_stokes[:, :, 0, :, :]  # [bs, c, h, w]
    S1 = filter_stokes[:, :, 1, :, :]
    S2 = filter_stokes[:, :, 2, :, :]
    S3 = filter_stokes[:, :, 3, :, :]
    
    # 计算HVADLR分量
    H = (S0 + S1) / 2
    V = (S0 - S1) / 2
    A = (S0 + S2) / 2
    D = (S0 - S2) / 2
    L = (S0 + S3) / 2
    R = (S0 - S3) / 2
    
    
    # 堆叠成 [bs, c, 6, h, w]
    filter_hvadlr = torch.stack([H, V, A, D, L, R], dim=2)
    
    return filter_hvadlr



# filter_matrix_test = stokes_filter_to_hvadlr_torch(filter_matrix_test)







# test_data = np.load('910_safe_raw.npy')
test_data = np.load('color_raw_new/12.7/7.npy')  # 70 100 100
# test_data = np.load('color_raw_new/reflect.npy')  # 70 100 100
test_data = np.load('color_raw_new/11.20/H.npy')


print(test_data.shape)
h, w = test_data.shape[0], test_data.shape[1]
filter_matrix_test = filter_matrix_test[:, 8, :, :h, :w]


input_data = test_data * 1 
# input_data = (test_data - np.min(test_data)) / (np.max(test_data) - np.min(test_data))
input_data = torch.tensor(input_data).cuda().float()
print(torch.mean(input_data))


test_input = input_data

with torch.no_grad():
    model_out = model(test_input, filter_matrix_test)
    # model_out = model_out.view(batch_size, 21, 4, 400, 700).squeeze(0)
    # model_out = model_out.view(batch_size, 21, 4, 300, 300).squeeze(0)
    # model_out = model_out.view(batch_size, 21, 4, 280, 380).squeeze(0)
    model_out = model_out.view(batch_size, 4, h, w).squeeze(0)
    # model_out = model_out.view(batch_size, 21, 4, 220, 300).squeeze(0)
    model_out = model_out.detach().cpu().numpy()



print(f'原始形状: {model_out.shape}')




save_path = f'raw_temp'
os.makedirs(save_path, exist_ok=True)
np.save(os.path.join(save_path, 'raw_full.npy'), model_out)



plt.subplot(2, 2, 1)
plt.imshow(model_out[0, :, :], cmap='gray')
plt.axis('off')
plt.title('S0')
plt.subplot(2, 2, 2)
plt.imshow(model_out[1, :, :], cmap='gray')
plt.axis('off')
plt.title('S1')
plt.subplot(2, 2, 3)
plt.imshow(model_out[2, :, :], cmap='gray')
plt.axis('off')
plt.title('S2')
plt.subplot(2, 2, 4)
plt.imshow(model_out[3, :, :], cmap='gray')
plt.axis('off')
plt.title('S3')
plt.savefig(os.path.join(save_path, 'raw_stokes.png'))
plt.close()












