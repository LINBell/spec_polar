import random

from torch import nn as nn
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
from option_dauhst import opt
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



batch_size = 6

if opt.method=='hdnet':
    model, FDL_loss = model_generator(opt.method, opt.pretrained_model_path).cuda()
else:
    model = model_generator(opt.method, opt.pretrained_model_path).cuda()

# model_path = 'mask/100_100_mask_epoch288_48.pth'
model_path = r'C:\Users\38362\Desktop\tio2_100_100_247_51.64.pth'
# model_path = 'polo_test_model/ma_attention.pth'
model.load_state_dict(torch.load(model_path))

filter_path = 'mask/tio2_100_mutual_100_100.npy'
filter_matrix = torch.from_numpy(np.load(filter_path)).unsqueeze(0).repeat(opt.batch_size, 1, 1, 1, 1).cuda().float()
filter_matrix_test = torch.from_numpy(np.load(filter_path)).unsqueeze(0).repeat(15, 1, 1, 1, 1).cuda().float()
filter_matrix = filter_matrix[:, :, :, :opt.crop_size, :opt.crop_size]
filter_matrix_test = filter_matrix_test[:, :, :, :opt.crop_size, :opt.crop_size]

# test_data = LoadTraining_npy_spec_polo('E:\\video_datasets\\spec_polo_test/')
test_data = LoadTraining_npy_spec_polo('F:\denoise_test')

test_data = np.stack(test_data, axis=0).transpose(0, 3, 4, 1, 2)  # 5 x 512 612 4 4



data_list = []
for i in range(1, 6):
    for j in range(1, 6):
        test_gt = test_data[:, :, :, 100*(i-1):100*i, 100*(j-1):100*j]

        real_gt = test_gt

        test_input = torch.tensor(test_gt).cuda()

        Phi_batch = filter_matrix_test
        Phi_s_batch = torch.sum(Phi_batch ** 2, [1,2])
        Phi_s_batch[Phi_s_batch == 0] = 1
        input_mask = [Phi_batch, Phi_s_batch]
        produtct = test_input * filter_matrix_test
        test_input = produtct.sum(dim=(1,2))


        # input_meas = init_meas(test_gt, mask3d_batch_test, opt.input_setting)
        model.eval()

        with torch.no_grad():
            model_out = model(test_input, input_mask)
            model_out = model_out.view(15, 21, 4, 100, 100)
        data_list.append(model_out)




grid = [[None for _ in range(5)] for _ in range(5)]

index = 0
for i in range(5):
    for j in range(5):
        grid[i][j] = data_list[index]
        index += 1

# 现在，我们需要沿着第3和第4维度（高度和宽度）进行拼接
# 首先沿着j方向（宽度）拼接每一行
rows = []
for i in range(5):
    # 将第i行的5个块沿着宽度（dim=4）拼接
    row_blocks = [grid[i][j] for j in range(5)]
    concatenated_row = torch.cat(row_blocks, dim=4)  # 沿宽度拼接
    rows.append(concatenated_row)

# 然后沿着i方向（高度）拼接所有行
full_image = torch.cat(rows, dim=3)  # 沿高度拼接  5 21 4 500 500


real_gt = test_data[:, :, :, :500, :500]

model_out = full_image.detach().cpu().numpy()

for i in range(1, 15):

    index = i

    gt_disp = real_gt[index, :, 0, :, :]
    out_disp = model_out[index, :, 0, :, :]

    psnr= np_psnr(out_disp, gt_disp)
    print(f'psnr:{np_psnr(out_disp, gt_disp)}')



    dop_gt = cal_dop(real_gt[index, :, :, :, :])
    dop_out = cal_dop(model_out[index, :, :, :, :])
    # 逐通道计算误差
    mae_per_channel = np.mean(np.abs(dop_out - dop_gt), axis=(1, 2))  # [c]
    rmse_per_channel = np.sqrt(np.mean((dop_out - dop_gt) ** 2, axis=(1, 2)))  # [c]

    # 全局平均误差（跨所有通道和像素）
    mae_global = np.mean(mae_per_channel)
    rmse_global = np.mean(rmse_per_channel)
    relative_error = np.mean(np.abs(dop_out - dop_gt) / (dop_gt + 1e-6))  # 避免除以零



    save_path = rf'C:\Users\38362\Desktop\mask_pic\dauhst\dauhst_100_100_247_51.64\{index}_{psnr:.2f}_MAE{mae_global:.4f}_RMSE{rmse_global:.4f}_avg{relative_error:.4f}'
    os.makedirs(save_path, exist_ok=True)

    gt_disp = stack_21(real_gt[index, :, 0, :, :].transpose(1, 2, 0))
    out_disp = stack_21(model_out[index, :, 0, :, :].transpose(1, 2, 0))

    # noise_scale = 1  # 噪声强度（标准差）
    # noise = np.random.normal(loc=0, scale=noise_scale, size=out_disp.shape).astype(np.float32)
    # out_disp = out_disp + noise


    plt.subplot(1, 2, 1)
    plt.imshow(gt_disp)
    plt.title('gt')
    plt.subplot(1, 2, 2)
    plt.imshow(out_disp*1)
    plt.title('reconstructed')
    plt.savefig(os.path.join(save_path, '13.png'))
    plt.show()





    gt_disp = real_gt[index, 0, 0, :, :]
    out_disp = model_out[index, 0, 0, :, :]
    gt_disp = (gt_disp-np.min(gt_disp)) / (np.max(gt_disp)-np.min(gt_disp))
    out_disp = (out_disp-np.min(out_disp)) / (np.max(out_disp)-np.min(out_disp))




    plt.subplot(1, 2, 1)
    plt.imshow(gt_disp, cmap='gray')
    plt.title('gt')
    plt.subplot(1, 2, 2)
    plt.imshow(out_disp*1,  cmap='gray')
    plt.title('reconstructed')
    plt.savefig(os.path.join(save_path, '0.png'))
    plt.show()

    def normalize_to_neg1_pos1(data):
        data_min, data_max = np.min(data), np.max(data)
        return 2 * (data - data_min) / (data_max - data_min) - 1

    channel_index = 0
    gt_disp = real_gt[index, channel_index, 3, :, :]  # min -0.01 max 0.015
    # gt_disp /= np.max(np.abs(gt_disp))
    print(f'polo real mean:{np.mean(gt_disp)}')
    print(f'polo real max:{np.max(gt_disp)}')
    print(f'polo real min:{np.min(gt_disp)}')
    out_disp = model_out[index, channel_index, 3, :, :]  # min -0.2 max 0.09
    print(f'polo out mean:{np.mean(out_disp)}')
    print(f'polo out max:{np.max(out_disp)}')
    print(f'polo out min:{np.min(out_disp)}')
    # gt_disp = normalize_to_neg1_pos1(gt_disp)
    # out_disp = normalize_to_neg1_pos1(out_disp)

    plt.subplot(1, 2, 1)
    plt.imshow(gt_disp, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(out_disp, cmap='gray')
    plt.savefig(os.path.join(save_path, '1.png'))
    plt.show()


    red_blue_cmap = plt.cm.bwr  # 使用 Matplotlib 自带的 bwr 颜色映射

    # noise_scale = 0.002  # 噪声强度（标准差）
    # noise = np.random.normal(loc=0, scale=noise_scale, size=out_disp.shape).astype(np.float32)
    # out_disp = out_disp + noise


    plt.subplot(1, 2, 1)
    plt.imshow(gt_disp*5, cmap=red_blue_cmap, vmin=-np.max(np.abs(gt_disp)), vmax=np.max(np.abs(gt_disp)))
    plt.colorbar()
    plt.axis('off')
    plt.title('gt')
    print(np.max(np.abs(gt_disp)))
    print(np.max(gt_disp))
    print(np.min(gt_disp))

    plt.subplot(1, 2, 2)
    plt.imshow(out_disp*5, cmap=red_blue_cmap, vmin=-np.max(np.abs(out_disp)), vmax=np.max(np.abs(out_disp)))
    plt.colorbar()
    plt.axis('off')
    plt.title('reconstructed')
    print(-np.max(np.abs(out_disp)))

    plt.savefig(os.path.join(save_path, '2_3.png'))
    plt.show()



    gt_disp = real_gt[index, channel_index, 1, :, :]
    out_disp = model_out[index, channel_index, 1, :, :]  # min -0.2 max 0.09
    red_blue_cmap = plt.cm.bwr  # 使用 Matplotlib 自带的 bwr 颜色映射

    plt.subplot(1, 2, 1)
    plt.imshow(gt_disp*5, cmap=red_blue_cmap, vmin=-np.max(np.abs(gt_disp)), vmax=np.max(np.abs(gt_disp)))
    plt.colorbar()
    plt.axis('off')
    plt.title('gt')
    print(np.max(np.abs(gt_disp)))
    print(np.max(gt_disp))
    print(np.min(gt_disp))

    plt.subplot(1, 2, 2)
    plt.imshow(out_disp*5, cmap=red_blue_cmap, vmin=-np.max(np.abs(out_disp)), vmax=np.max(np.abs(out_disp)))
    plt.colorbar()
    plt.axis('off')
    plt.title('reconstructed')
    print(-np.max(np.abs(out_disp)))

    plt.savefig(os.path.join(save_path, '2_1.png'))
    plt.show()

    gt_disp = real_gt[index, channel_index, 2, :, :]
    out_disp = model_out[index, channel_index, 2, :, :]  # min -0.2 max 0.09
    red_blue_cmap = plt.cm.bwr  # 使用 Matplotlib 自带的 bwr 颜色映射

    plt.subplot(1, 2, 1)
    plt.imshow(gt_disp*5, cmap=red_blue_cmap, vmin=-np.max(np.abs(gt_disp)), vmax=np.max(np.abs(gt_disp)))
    plt.colorbar()
    plt.axis('off')
    plt.title('gt')
    print(np.max(np.abs(gt_disp)))
    print(np.max(gt_disp))
    print(np.min(gt_disp))

    plt.subplot(1, 2, 2)
    plt.imshow(out_disp*5, cmap=red_blue_cmap, vmin=-np.max(np.abs(out_disp)), vmax=np.max(np.abs(out_disp)))
    plt.colorbar()
    plt.axis('off')
    plt.title('reconstructed')
    print(-np.max(np.abs(out_disp)))

    plt.savefig(os.path.join(save_path, '2_2.png'))
    plt.show()








    gt_disp = real_gt[index, :, 3, :, :]  # 21 h w
    out_disp = model_out[index, :, 3, :, :] # 21 h w
    c, h, w = gt_disp.shape
    select_h = random.randint(0, h-1)
    select_w = random.randint(0, w-1)
    gt_disp = gt_disp[:, select_h, select_w]
    out_disp = out_disp[:, select_h, select_w]
    x = np.linspace(0, 20, 21)
    plt.plot(x, gt_disp, label='gt', color='green', linewidth=2)
    plt.plot(x, out_disp, label='recons', color='orange', linewidth=2)
    plt.legend()
    plt.savefig(os.path.join(save_path, '14.png'))
    plt.show()







    gt_disp = real_gt[index, :, :, :, :].transpose(2, 3, 0, 1)
    out_disp = model_out[index, :, :, :, :].transpose(2, 3, 0, 1)
    print('GT:::::::::::')
    cal_restric(gt_disp)
    print('out:::::::::::')
    cal_restric(out_disp)








    def plot_polarization_ellipse(S1, S2, S3, scale=1):
        """绘制单个偏振椭圆"""
        psi = 0.5 * np.arctan2(S2, S1)  # 方位角
        DoP = np.sqrt(S1 ** 2 + S2 ** 2 + S3 ** 2)  # 偏振度
        chi = 0.5 * np.arcsin(S3 / (DoP + 1e-9))  # 椭圆率（避免除零）

        # 椭圆参数
        a = DoP * scale  # 长轴
        b = a * np.tan(chi)  # 短轴

        # 生成椭圆点
        theta = np.linspace(0, 2 * np.pi, 100)
        x = a * np.cos(theta) * np.cos(psi) - b * np.sin(theta) * np.sin(psi)
        y = a * np.cos(theta) * np.sin(psi) + b * np.sin(theta) * np.cos(psi)

        plt.plot(x, y, 'r-', linewidth=1)  # 绘制椭圆
        plt.axhline(0, color='k', linestyle='--', linewidth=0.5)  # 坐标轴
        plt.axvline(0, color='k', linestyle='--', linewidth=0.5)
        plt.axis('equal')


    gt_disp = real_gt[index, 0, :, :, :].transpose(1, 2, 0)
    out_disp = model_out[index, 0, :, :, :].transpose(1, 2, 0)

    # 示例：绘制某个像素的偏振椭圆
    S1, S2, S3 = gt_disp[250, 250, 1], gt_disp[250, 250, 2], gt_disp[250, 250, 3]  # 中心像素
    plot_polarization_ellipse(S1, S2, S3)
    plt.title("Polarization Ellipse (Reconstructed)")

    plt.savefig(os.path.join(save_path, '11.png'))
    plt.show()

    S1, S2, S3 = out_disp[250, 250, 1], out_disp[250, 250, 2], out_disp[250, 250, 3]  # 中心像素
    plot_polarization_ellipse(S1, S2, S3)
    plt.title("Polarization Ellipse (Reconstructed)")

    plt.savefig(os.path.join(save_path, '12.png'))
    plt.show()


    for i in range(3, 11):
        select_h = random.randint(0,499)
        select_w = random.randint(0,499)
        gt_disp = real_gt[index, :, 0, select_h, select_w]
        out_disp = model_out[index, :, 0, select_h, select_w]
        x = np.array(list(range(21)))
        plt.figure()
        plt.plot(x, gt_disp, label='gt', marker='o', color='red')
        plt.plot(x, out_disp, label='reconstructed', marker='s', color='blue')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend()
        plt.title('Line Plot of Two [4]-shaped Data')
        plt.savefig(os.path.join(save_path, f'{i}.png'))
        # plt.savefig(f'{i}.png')



    out_disp = model_out[index, channel_index, :, :, :].transpose(1, 2, 0)
    gt_disp = real_gt[index, channel_index, :, :, :].transpose(1, 2, 0)

    # noise_scale = 0.005  # 噪声强度（标准差）
    # noise = np.random.normal(loc=0, scale=noise_scale, size=out_disp.shape).astype(np.float32)
    # out_disp = out_disp + noise


    stokes_data = out_disp
    # 提取斯托克斯参数
    S0 = stokes_data[..., 0]  # 总光强
    S1 = stokes_data[..., 1]  # 线偏振分量1
    S2 = stokes_data[..., 2]  # 线偏振分量2
    S3 = stokes_data[..., 3]  # 圆偏振分量

    # 计算方位角（直接保留 -90°~90° 范围）
    azimuth_deg = 0.5 * np.rad2deg(np.arctan2(S2, S1))  # 范围: [-90°, 90°]

    # 计算偏振度（DoP）
    DoP = np.sqrt(S1**2 + S2**2 + S3**2) / (S0 + 1e-10)  # 避免除零

    # 可视化
    plt.figure(figsize=(12, 5))

    # 1. 方位角图（HSV 色彩映射，范围 -90°~90°）
    plt.subplot(1, 2, 1)
    hue = (azimuth_deg + 90) / 180.0  # 将 -90°~90° 映射到 0~1（HSV 的 H 通道）
    saturation = np.ones_like(hue)     # 饱和度=1
    value = np.ones_like(hue)         # 亮度=1
    hsv_image = np.stack([hue, saturation, value], axis=-1)
    rgb_image = colors.hsv_to_rgb(hsv_image)
    plt.imshow(rgb_image)
    plt.title("Azimuth Angle (HSV, -90°~90°)")
    plt.colorbar(ticks=[-90, -45, 0, 45, 90], label="Angle (°)")

    # 2. 偏振度图（灰度）
    plt.subplot(1, 2, 2)
    plt.imshow(DoP, cmap='gray', vmin=0, vmax=1)
    plt.title("Degree of Polarization (DoP)")
    plt.colorbar(label="DoP (0~1)")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '15.png'))
    plt.show()



    stokes_data = gt_disp
    # 提取斯托克斯参数
    S0 = stokes_data[..., 0]  # 总光强
    S1 = stokes_data[..., 1]  # 线偏振分量1
    S2 = stokes_data[..., 2]  # 线偏振分量2
    S3 = stokes_data[..., 3]  # 圆偏振分量

    # 计算方位角（直接保留 -90°~90° 范围）
    azimuth_deg = 0.5 * np.rad2deg(np.arctan2(S2, S1))  # 范围: [-90°, 90°]

    # 计算偏振度（DoP）
    DoP = np.sqrt(S1**2 + S2**2 + S3**2) / (S0 + 1e-10)  # 避免除零

    # 可视化
    plt.figure(figsize=(12, 5))

    # 1. 方位角图（HSV 色彩映射，范围 -90°~90°）
    plt.subplot(1, 2, 1)
    hue = (azimuth_deg + 90) / 180.0  # 将 -90°~90° 映射到 0~1（HSV 的 H 通道）
    saturation = np.ones_like(hue)     # 饱和度=1
    value = np.ones_like(hue)         # 亮度=1
    hsv_image = np.stack([hue, saturation, value], axis=-1)
    rgb_image = colors.hsv_to_rgb(hsv_image)
    plt.imshow(rgb_image)
    plt.title("Azimuth Angle (HSV, -90°~90°)")
    plt.colorbar(ticks=[-90, -45, 0, 45, 90], label="Angle (°)")

    # 2. 偏振度图（灰度）
    plt.subplot(1, 2, 2)
    plt.imshow(DoP, cmap='gray', vmin=0, vmax=1)
    plt.title("Degree of Polarization (DoP)")
    plt.colorbar(label="DoP (0~1)")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '16.png'))
    plt.show()