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
from collections import Counter

os.environ["CUDA_VISIBLE_DEVICES"] = '2'



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


model_path = '/data8T/lzj/MST_Spectral_remote/simulation/train_code/GEN/gen_large/model/model_epoch_108.pth'
model.load_state_dict(torch.load(model_path))


filter_path = 'mask/93_hand_bd_stokes.npy'
filter_path = 'mask/190_190_gen_data_train_curve.npy'
filter_matrix = torch.from_numpy(np.load(filter_path)).unsqueeze(0).repeat(opt.batch_size, 1, 1, 1, 1).cuda().float()
filter_matrix_test = torch.from_numpy(np.load(filter_path)).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1).cuda().float()
filter_matrix = filter_matrix[:, :, :, :opt.crop_size, :opt.crop_size]
filter_matrix_test = filter_matrix_test[:, :, :, :opt.crop_size, :opt.crop_size]

# test_data = ds_LoadValidation_npy_spec_polo(opt.test_path, means, stds, states=True)

# test_data = np.stack(test_data, axis=0).transpose(0, 3, 4, 1, 2)  # 5 x 512 612 4 4

file = '99_H_spec_raw'
# file = '99_V_spec_raw'
for filename in os.listdir(file):
    if filename.endswith('npy'):
        raw_name = filename
        cur_raw_path = os.path.join(file, filename)

        # data_list = []
        # raw_name = '630_V_raw'


        # test_data = np.load('550_H_raw.npy')
        test_data = np.load(cur_raw_path)
        print(f'Now Processing {cur_raw_path}')
        # test_data /=  2

        # test_data = (test_data - np.min(test_data)) / (np.max(test_data) - np.min(test_data))

        # for i in range(256):
            # print(i)
        input_data = test_data[:, :]
        input_data = torch.tensor(input_data).cuda().float()

        # print(input_data.shape)
        model.eval()
        with torch.no_grad():
            model_out = model(input_data, filter_matrix_test)
            model_out = model_out.view(batch_size, 21, 4, opt.crop_size, opt.crop_size).squeeze(0)
            model_out = model_out.detach().cpu().numpy()  # 21 4 116 116




        save_path = f'duizhao/gen_large_108_H/{raw_name}'
        os.makedirs(save_path, exist_ok=True)

        channel_index = [0, 10, 20]
        wavelengths = np.arange(450, 651, 10)

        dop, docp = cal_deg(model_out)

        avg_dop = np.mean(dop)
        avg_docp = np.mean(docp)
        print(f"全局平均偏振度 (DoP): {avg_dop:.4f}")
        print(f"全局平均圆偏振度 (DoCP): {avg_docp:.4f}")

        # 先计算全局的峰值统计（在循环外部）
        # 假设 model_out 形状为 [channels, states, height, width]
        max_indices = np.argmax(model_out, axis=0)  # 形状变为 [states, height, width]
        max_wavelengths = wavelengths[max_indices]  # 形状为 [states, height, width]

        # 对于每个状态，统计最常见的峰值
        for i in range(4):  # 遍历4个状态
            # 获取当前状态的峰值波长分布
            state_max_wavelengths = max_wavelengths[i]  # 形状为 [height, width]
            
            # 统计当前状态的最常见峰值波长
            all_wavelengths = state_max_wavelengths.flatten()
            wavelength_counter = Counter(all_wavelengths)
            most_common_wavelength, count = wavelength_counter.most_common(1)[0]
            total_pixels = len(all_wavelengths)  # 总像素数
            proportion = count / total_pixels  # 最常见波长所占比例
            if i == 0:
                print(f"  最常见峰值波长: {most_common_wavelength}")
                print(f"  所占比例: {proportion:.2%}")

            # 找到所有满足条件的像素
            matching_pixels_mask = (state_max_wavelengths == most_common_wavelength)
            matching_pixels_coords = np.argwhere(matching_pixels_mask)
            
            # 随机选择5个点或者选择前5个点
            for index in range(min(5, len(matching_pixels_coords))):
                y, x = matching_pixels_coords[index]  # 注意：np.argwhere 返回的是 (y, x)
                
                # 获取当前点的光谱
                point_spectrum = model_out[:, i, y, x]  # 所有通道在当前状态和位置的光谱
                
                # 找到当前光谱的峰值（用于绘图）
                max_idx_point = np.argmax(point_spectrum)
                max_wavelength_point = wavelengths[max_idx_point]
                
                # 获取当前的DoP和DoCP（注意索引顺序）
                cur_dop = dop[max_idx_point, y, x]  # 假设dop形状为 [states, height, width]
                cur_docp = docp[max_idx_point, y, x]  # 假设docp形状为 [states, height, width]
                
                plt.figure(figsize=(10, 6))
                plt.plot(wavelengths, point_spectrum, 'b-', linewidth=2)
                plt.xlabel('Wavelength (nm)', fontsize=12)
                plt.ylabel('Intensity', fontsize=12)
                plt.title(f'State {i}, Position ({x}, {y}), DoP {cur_dop:.2f}, DoCP {cur_docp:.2f}, avg_dop:{avg_dop:.2f}, avg_docp{avg_docp:.4f}', fontsize=14)
                plt.grid(True, alpha=0.3)
                
                # 画最大值垂直线
                plt.axvline(x=max_wavelength_point, color='red', linestyle='--', linewidth=2, alpha=0.8)
                
                # 在x轴下方添加波长值标注
                y_min, y_max = plt.ylim()
                plt.text(max_wavelength_point, y_min - (y_max - y_min) * 0.05, 
                        f'{max_wavelength_point}nm', 
                        fontsize=10, color='red', ha='center', va='top')
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, f'State{i}_Pos{x}_{y}.png'), dpi=300, bbox_inches='tight')
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