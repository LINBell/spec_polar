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

# os.environ["CUDA_VISIBLE_DEVICES"] = '2'



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


model_path = '9_9/basic/model/model_epoch_269.pth'
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
# test_data = np.load('913_safe_raw.npy')
test_data = np.load('color_raw/1_3w_blue.npy')
# test_data = np.load('color_raw/2_3w_red.npy')
# test_data = np.load('color_raw/1_3w_green.npy')
# test_data *= 0.5


# input_data = test_data * 5
input_data = (test_data - np.min(test_data)) / (np.max(test_data) - np.min(test_data))
input_data = torch.tensor(input_data).cuda().float()
# input_data = torch.tensor(test_data).cuda().float()


# print(input_data.shape)
model.eval()
with torch.no_grad():
    model_out = model(input_data, filter_matrix_test)
    model_out = model_out.view(batch_size, 21, 4, opt.crop_size, opt.crop_size).squeeze(0)
    model_out = model_out.detach().cpu().numpy()






save_path = f'raw_temp'
os.makedirs(save_path, exist_ok=True)
np.save(os.path.join(save_path, 'raw_full.npy'), model_out)
channel_index = [0, 10, 20]

# rgb_st, rgb = trans_multi_2_rgb(model_out[:, 0, :, :].transpose(1, 2, 0))
rgb = trans_multi_2_rgb_v2(model_out[:, 0, :, :].transpose(1, 2, 0))

# np.save(os.path.join(save_path, 'raw.npy'), final_image_full[channel_index, 0, :, :].transpose(1, 2, 0))

np.save(os.path.join(save_path, 'raw.npy'), rgb)
plt.figure()
# plt.imshow(final_image_full[0, 0, :, :].transpose(1, 2, 0), cmap='gray')
plt.subplot(1, 2, 1)
plt.imshow(rgb*1)
plt.axis('off')
plt.subplot(1, 2, 2)
# plt.imshow(rgb_st)
plt.imshow(rgb)
plt.axis('off')
plt.savefig(os.path.join(save_path, 'raw.png'))
plt.close()








# for i in range(21):
#     save_path = f'/data4T/lzj/mst_spectral/simulation/train_code/exp/norm_denorm/save_pic/raw/{i}'
#     os.makedirs(save_path, exist_ok=True)
#     index = i

#     img = (data_list[i, :, 0, :, :] + data_list[i, :, 1, :, :]) / 2  # 21 1 100 100
#     print(img.shape) # 21 100 100 


#     c, h, w = img.shape

#     # 随机选取5个点的坐标
#     points = []
#     for _ in range(5):
#         x = random.randint(0, h-1)
#         y = random.randint(0, w-1)
#         points.append((x, y))



#     # 每个点画一张图
#     for idx, (x, y) in enumerate(points):
#         pred_curve = img[:, x, y]
#         plt.figure(figsize=(6, 4))
#         plt.plot(range(c), pred_curve, label='Pred')
#         plt.title(f'点({x},{y})')
#         plt.xlabel('波段')
#         plt.ylabel('强度')
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(os.path.join(save_path, f'random_point_{idx+1}_spectrum.png'))
#         plt.show()