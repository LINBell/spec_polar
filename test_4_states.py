import random

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

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


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


batch_size = 15


model = model_generator('spec_polo', opt.pretrained_model_path).cuda()


model_path = '/data8T/lzj/MST_Spectral_remote/simulation/train_code/exp/mst_l/4_states_bd_mask/model/model_epoch_194.pth'
model.load_state_dict(torch.load(model_path))


filter_path = '/data8T/lzj/MST_Spectral_remote/simulation/train_code/mask/tio2_100_mutual_10_10.npy'
filter_matrix = torch.from_numpy(np.load(filter_path)).unsqueeze(0).repeat(opt.batch_size, 1, 1, 1, 1).cuda().float()
filter_matrix_test = torch.from_numpy(np.load(filter_path)).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1).cuda().float()
filter_matrix = filter_matrix[:, :, :, :opt.crop_size, :opt.crop_size]
filter_matrix_test = filter_matrix_test[:, :, :, :opt.crop_size, :opt.crop_size]

test_data = LoadTraining_npy_spec_polo('/data8T/lzj/Spectral_data/denoised/hyperspectral/test', states=True)

test_data = np.stack(test_data, axis=0).transpose(0, 3, 4, 1, 2)  # 5 x 512 612 4 4




data_list = []
for i in range(1, 6):
    for j in range(1, 6):
        test_gt = test_data[:, :, :, 100*(i-1):100*i, 100*(j-1):100*j]
        real_gt = test_gt
        test_input = torch.tensor(test_gt).cuda()
        model.eval()

        with torch.no_grad():
            model_out = model(test_input, filter_matrix_test)
            model_out = model_out.view(batch_size, 21, 4, 100, 100)
        data_list.append(model_out)




grid = [[None for _ in range(5)] for _ in range(5)]
index = 0
for i in range(5):
    for j in range(5):
        grid[i][j] = data_list[index]
        index += 1
rows = []
for i in range(5):
    row_blocks = [grid[i][j] for j in range(5)]
    concatenated_row = torch.cat(row_blocks, dim=4)  # 沿宽度拼接
    rows.append(concatenated_row)
full_image = torch.cat(rows, dim=3)  




real_gt = test_data[:, :, :, :500, :500]

model_out = full_image.detach().cpu().numpy()

for i in range(1, 15):
    index = i
    gt_disp = real_gt[index, :, 0, :, :]
    out_disp = model_out[index, :, 0, :, :]
    psnr= np_psnr(out_disp, gt_disp)
    print(f'psnr:{np_psnr(out_disp, gt_disp)}')




    save_path = f'/data8T/lzj/save_pic/V1/{index}_{psnr:.2f}'
    os.makedirs(save_path, exist_ok=True)

    gt_disp = stack_21(real_gt[index, :, 0, :, :].transpose(1, 2, 0))
    out_disp = stack_21(model_out[index, :, 0, :, :].transpose(1, 2, 0))
    plt.subplot(1, 2, 1)
    plt.imshow(gt_disp)
    plt.title('gt')
    plt.subplot(1, 2, 2)
    plt.imshow(out_disp*1)
    plt.title('reconstructed')
    plt.savefig(os.path.join(save_path, '13.png'))
    plt.show()

