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


model = model_generator('spec_polo_test', opt.pretrained_model_path).cuda()


model_path = '9_9/basic/model/model_epoch_265.pth'
model_path = '9_9/six_shuffle/model/model_epoch_212.pth'
# model_path = '/data4T/lzj/mst_spectral/simulation/train_code/duizhao/zero/model/model_epoch_51.pth'
model.load_state_dict(torch.load(model_path))


filter_path = 'mask/99_hand_bd_matrix.npy'
filter_path = 'mask/99_100_100_full_matrix.npy'
filter_matrix = torch.from_numpy(np.load(filter_path)).unsqueeze(0).repeat(opt.batch_size, 1, 1, 1, 1).cuda().float()
filter_matrix_test = torch.from_numpy(np.load(filter_path)).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1).cuda().float()
filter_matrix = filter_matrix[:, :, :, :opt.crop_size, :opt.crop_size]
filter_matrix_test = filter_matrix_test[:, :, :, :opt.crop_size, :opt.crop_size]

# test_data = ds_LoadValidation_npy_spec_polo(opt.test_path, means, stds, states=True)

# test_data = np.stack(test_data, axis=0).transpose(0, 3, 4, 1, 2)  # 5 x 512 612 4 4
data_list = []

# test_data = np.load('910_safe_raw.npy')
test_data = np.load('color_raw/color_raw.npy')  # 70 100 100
# test_data = np.load('color_raw/color_3w_1_raw.npy')  # 70 100 100
# test_data = np.load('color_raw/color_2w_2_raw.npy')  # 70 100 100
test_data = np.load('color_raw/color_2w_2_raw.npy')  # 70 100 100
test_data = np.load('color_raw/fangshe_2w_2.npy')  # 70 100 100
test_data = np.load('color_raw/color_1w.npy')  # 70 100 100
test_data = np.load('color_raw/10x_3w.npy')  # 70 100 100




input_data = test_data * 1 
input_data = (test_data - np.min(test_data)) / (np.max(test_data) - np.min(test_data))
input_data = torch.tensor(input_data).cuda().float()
print(torch.mean(input_data))

data_list = []

for i in range(9):
# for i in range(8):
# for i in range(6):
# for i in range(4):
    test_input = input_data[i, :, :]
    # test_input = (test_input - torch.min(test_input)) / (torch.max(test_input) - torch.min(test_input))
    with torch.no_grad():
        model_out = model(test_input, filter_matrix_test)
        model_out = model_out.view(batch_size, 21, 4, opt.crop_size, opt.crop_size).squeeze(0)
        model_out = model_out.detach().cpu().numpy()
    data_list.append(model_out)


out_put = np.stack(data_list, axis=0)
print(f'原始形状: {out_put.shape}')

# 首先将第一个维度(70)和最后两个维度(100,100)合并
# 重塑为 (7, 10, 21, 4, 100, 100)
reshaped = out_put.reshape(3, 3, 21, 4, 100, 100)
# reshaped = out_put.reshape(2, 4, 21, 4, 100, 100)
# reshaped = out_put.reshape(2, 3, 21, 4, 100, 100)
# reshaped = out_put.reshape(2, 2, 21, 4, 100, 100)

# 交换轴，将空间维度合并
# 从 (7, 10, 21, 4, 100, 100) 到 (21, 4, 7, 100, 10, 100)
swapped = reshaped.transpose(2, 3, 0, 4, 1, 5)

# 最终重塑为 (21, 4, 700, 1000)
result = swapped.reshape(21, 4, 300,300)
# result = swapped.reshape(21, 4, 200, 400)
# result = swapped.reshape(21, 4, 200, 300)
# result = swapped.reshape(21, 4, 200, 200)

print(f'最终形状: {result.shape}')  # (21, 4, 700, 1000)


model_out = result




save_path = f'raw_temp'
os.makedirs(save_path, exist_ok=True)
np.save(os.path.join(save_path, 'raw_full.npy'), model_out)
channel_index = [0, 10, 20]

# rgb_st, rgb = trans_multi_2_rgb(model_out[:, 0, :, :].transpose(1, 2, 0))
rgb = trans_multi_2_rgb_v2(model_out[:, 0, :, :].transpose(1, 2, 0))
# rgbs, rgb = trans_multi_2_rgb(model_out[:, 0, :, :].transpose(1, 2, 0))

# np.save(os.path.join(save_path, 'raw.npy'), final_image_full[channel_index, 0, :, :].transpose(1, 2, 0))

np.save(os.path.join(save_path, 'raw.npy'), rgb)
plt.figure()
# plt.imshow(final_image_full[0, 0, :, :].transpose(1, 2, 0), cmap='gray')
plt.subplot(1, 2, 1)
plt.imshow(rgb*10)
plt.axis('off')
plt.subplot(1, 2, 2)
# plt.imshow(rgb_st)
plt.imshow(rgb)
plt.axis('off')
plt.savefig(os.path.join(save_path, 'raw.png'))
plt.close()









# rgb_list = []  # 用于存储RGB转换后的小块

# for i in range(70):
#     test_input = input_data[i, :, :]
#     test_input = (test_input - torch.min(test_input)) / (torch.max(test_input) - torch.min(test_input))
#     with torch.no_grad():
#         model_out = model(test_input, filter_matrix_test)
#         model_out = model_out.view(batch_size, 21, 4, opt.crop_size, opt.crop_size).squeeze(0)
#         model_out = model_out.detach().cpu().numpy()
    
#     # 只提取偏振维度的第0个维度（光强）
#     # model_out形状为 (21, 4, 100, 100)，取第0个偏振维度
#     intensity_data = model_out[:, 0, :, :]  # 形状为 (21, 100, 100)
    
#     # 转置为 (100, 100, 21) 以适应 trans_multi_2_rgb 函数
#     hs_data = intensity_data.transpose(1, 2, 0)
    
#     # 进行高光谱到RGB转换
#     rgb_small = trans_multi_2_rgb_v2(hs_data)
    
#     # 存储RGB结果，形状为 (100, 100, 3)
#     rgb_list.append(rgb_small)

# # 将RGB小块堆叠成完整图像
# # 首先将列表转换为数组，形状为 (70, 100, 100, 3)
# rgb_array = np.stack(rgb_list, axis=0)

# print(f'RGB小块形状: {rgb_array.shape}')

# # 重塑为 (7, 10, 100, 100, 3)
# reshaped = rgb_array.reshape(7, 10, 100, 100, 3)

# # 交换轴，将空间维度合并
# # 从 (7, 10, 100, 100, 3) 到 (3, 7, 100, 10, 100)
# swapped = reshaped.transpose(4, 0, 2, 1, 3)

# # 最终重塑为 (3, 700, 1000)
# result_rgb = swapped.reshape(3, 700, 1000)

# print(f'最终RGB形状: {result_rgb.shape}')  # (3, 700, 1000)

# # 转置为更常见的图像格式 (700, 1000, 3)
# final_rgb_image = result_rgb.transpose(1, 2, 0)

# save_path = f'raw_temp'
# os.makedirs(save_path, exist_ok=True)

# # 保存RGB图像
# np.save(os.path.join(save_path, 'raw_rgb.npy'), final_rgb_image)

# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.imshow(final_rgb_image)  # 调整亮度
# plt.axis('off')
# plt.title('RGB Image (Intensity)')

# plt.subplot(1, 2, 2)
# plt.imshow(final_rgb_image * 2)  # 不同的亮度调整
# plt.axis('off')
# plt.title('RGB Image (Adjusted)')

# plt.savefig(os.path.join(save_path, 'raw_rgb.png'))
# plt.close()

# print("处理完成：先转换小块为RGB，再拼接完整图像")








