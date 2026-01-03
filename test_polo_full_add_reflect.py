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
from option import opt
import torch.nn.functional as F


result_path = 'temp_val1'+ '/' + '/result/'  # train_t_mask 为1_1


if not os.path.exists(result_path):
    os.makedirs(result_path)

name = result_path 



def trans_multi_2_rgb(hyperspectral_data):
    # h w c
    cie_wavelength = np.array([450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 
                          550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650])

    # CIE 1931 XYZ色彩匹配函数值（对应上面的波长）
    cie_x = np.array([0.3362, 0.2908, 0.1954, 0.0956, 0.0320, 0.0049, 0.0093, 0.0633, 0.1655, 0.2904,
                    0.4334, 0.5945, 0.7621, 0.9163, 1.0263, 1.0622, 1.0026, 0.8544, 0.6424, 0.4479, 0.2835])

    cie_y = np.array([0.0380, 0.0600, 0.0910, 0.1390, 0.2080, 0.3230, 0.5030, 0.7100, 0.8620, 0.9540,
                    0.9950, 0.9950, 0.9520, 0.8700, 0.7570, 0.6310, 0.5030, 0.3810, 0.2650, 0.1750, 0.1070])

    cie_z = np.array([1.7721, 1.6692, 1.2876, 0.8130, 0.4652, 0.2720, 0.1582, 0.0782, 0.0422, 0.0203,
                    0.0087, 0.0039, 0.0021, 0.0017, 0.0011, 0.0008, 0.0003, 0.0002, 0.0000, 0.0000, 0.0000])
    
    your_wavelengths = np.arange(450, 651, 10)

    weights_r = cie_x / np.sum(cie_x)
    weights_g = cie_y / np.sum(cie_y)
    weights_b = cie_z / np.sum(cie_z)

    # print("红色通道权重:", weights_r)
    # print("绿色通道权重:", weights_g)
    # print("蓝色通道权重:", weights_b)

    # 6. 对高光谱数据的每个像素进行加权融合
    r_channel = np.tensordot(hyperspectral_data, weights_r, axes=([2], [0]))
    g_channel = np.tensordot(hyperspectral_data, weights_g, axes=([2], [0]))
    b_channel = np.tensordot(hyperspectral_data, weights_b, axes=([2], [0]))

    # 7. 堆叠通道并进行对比度拉伸
    rgb_image_CIE = np.stack([r_channel, g_channel, b_channel], axis=-1)
    rgb_stretched_CIE = np.stack([stretch_band(r_channel),
                                stretch_band(g_channel),
                                stretch_band(b_channel)], axis=-1)

    return rgb_stretched_CIE, rgb_image_CIE


def draw_test_images(truth, pred, path, psnr_list, ssim_list):  # bs h w c s
    channle_index = [0, 10, 20]
    if not os.path.exists(path):
        os.makedirs(path)
    bs, h, w, c, s = truth.shape
    # print(truth.shape)
    for index in range(bs):
        if index == 1:
            np.save(os.path.join(path, 'gt.npy'), truth[index, :, :, :, 0] )
            np.save(os.path.join(path, 'out.npy'), pred[index, :, :, :, 0] )
        # 取第10个波段的S0和S1，求平均
        gt = truth[index, :, :, :, 0] 
        out = pred[index, :, :, :, 0] 
        # print(f"gt shape after indexing: {gt.shape}")
        # print(f"out shape after indexing: {out.shape}")
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        # plt.imshow(gt.transpose(1, 2, 0))
        gt_disp, _ = trans_multi_2_rgb(gt)
        plt.imshow(gt_disp)
        plt.title('GT')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        # plt.imshow(out.transpose(1, 2, 0))
        out_disp, _ = trans_multi_2_rgb(out)
        plt.imshow(out_disp)
        plt.title('out')
        plt.axis('off')
        save_path = os.path.join(path, "{}_{:.2f}_{:.2f}.png".format(index, psnr_list[index], ssim_list[index]))
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=200)
        plt.close()


        for i in range(4):
            select_x = random.randint(0, h-1)
            select_y = random.randint(0, w-1)
            # 提取该像素点所有波段的光谱（S0和S1平均）
            gt_spec = truth[index, select_x, select_y, :, i] 
            out_spec = pred[index, select_x, select_y, :, i] 
            plt.figure()
            plt.plot(gt_spec, 'r', label='GT')
            plt.plot(out_spec, 'y', label='out')
            plt.title(f'Pixel ({select_x},{select_y},States{i}) Spectrum')
            plt.xlabel('Wavelength Index')
            plt.ylabel('Intensity')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(path, f"{index}_spec_{i}.png"), bbox_inches='tight', pad_inches=0.1, dpi=200)
            plt.close()

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

# def np_psnr(img, ref):  #  c s h w 4 4 200 200
#     img = (img * 256).round().clip(0, 255)
#     ref = (ref * 256).round().clip(0, 255)
#     nC, s = img.shape[0], img.shape[1]
#     psnr = 0
#     for i in range(nC):
#         for j in range(s):
#             mse = np.mean((img[i, j, :, :] - ref[i, j, :, :]) ** 2)
#             psnr += 10 * np.log10((255*255)/mse)
#     return psnr / (nC * s)
def np_psnr(img, ref):  #  c s h w 4 4 200 200
    img = (img * 256).round().clip(0, 255)
    ref = (ref * 256).round().clip(0, 255)
    nC= img.shape[0]
    psnr = 0
    for i in range(nC):
        mse = np.mean((img[i, :, : , :] - ref[i, :, :, :]) ** 2)
        psnr += 10 * np.log10((255*255)/mse)
    return psnr / (nC)

# scene_path = 'E:\\video_datasets\\311_spec_polo\\0000.npy'
#
# img = np.load(scene_path)
#
# img = img[:, :, :, [0, 7, 14, 20]].transpose(0, 1, 3, 2).astype(np.float32)
# img = img[:, :, :, [0, 7, 14, 20]].transpose(0, 1, 3, 2).astype(np.float32)
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

batch_size = 15

if opt.method=='hdnet':
    model, FDL_loss = model_generator(opt.method, opt.pretrained_model_path).cuda()
else:
    model = model_generator('spec_polo', opt.pretrained_model_path).cuda()

model_path = '9.3_hand_bd/10_10_v2/model/model_epoch_11.pth'
model_path = '9_9/add_reflect_denoise/model/model_epoch_212.pth'
# model_path = '9_9/basic/model/model_epoch_269.pth'
# model_path = '9_9/shuffle/model/model_epoch_171.pth'
# model_path = '9_9/six/model/model_epoch_212.pth'
# model_path = '9_9/shuffle_10/model/model_epoch_81.pth'
# model_path = 'polo_test_model/ma_attention.pth'
model.load_state_dict(torch.load(model_path))

# filter_path = 'transpose_matrix.pth'
# filter_matrix = torch.load(filter_path).unsqueeze(0).repeat(opt.batch_size, 1, 1, 1, 1).cuda()
# filter_matrix_test = torch.load(filter_path).unsqueeze(0).repeat(5, 1, 1, 1, 1).cuda()


# filter_path = '21_trans_matrix.npy'
# filter_path = 'self_made_matrix.npy'
# filter_path = 'select_21_matrix.npy'
# filter_path = 'select_21_matrix.npy'
filter_path = 'mask/93_hand_bd_10_10.npy'
filter_path = 'mask/99_100_100_full_matrix.npy' #更换为自己矩阵
# filter_path = 'mask/99_100_100_full_matrix_4T.npy' #更换为自己矩阵
# filter_path = 'mask/99_100_100_full_matrix_10T.npy' #更换为自己矩阵
filter_matrix = torch.from_numpy(np.load(filter_path)).unsqueeze(0).repeat(opt.batch_size, 1, 1, 1, 1).cuda().float()
filter_matrix_test = torch.from_numpy(np.load(filter_path)).unsqueeze(0).repeat(1, 1, 1, 7, 7).cuda().float()
filter_matrix = filter_matrix[:, :, :, :opt.crop_size, :opt.crop_size]
# filter_matrix_test = filter_matrix_test[:, :, :, :opt.crop_size, :opt.crop_size]



img_num = 15

# test_data = LoadTraining_npy_spec_polo('E:\\video_datasets\\spec_polo_test/')
# test_data = LoadTraining_npy_spec_polo(opt.data_path, norm=True)
# test_data_add_reflect = random_load_spec_polo(opt.data_path, max_images=img_num, norm=True, add_reflect=True)
test_data_add_reflect = random_load_spec_polo(opt.test_path, max_images=img_num, norm=True, add_reflect=True)
# test_data_add_reflect = LoadTraining_npy_spec_polo(opt.test_path, norm=True)

test_data_add_reflect = np.stack(test_data_add_reflect, axis=0).transpose(0, 3, 4, 1, 2)  # 5 x 512 612 4 4

# for i in range(15):
#     data_temp = test_data[i, :, 0, :, :]
#     print(np.min(data_temp))
#     print(np.max(data_temp))
#     print(np.mean(data_temp))

psnr_list, ssim_list = [], []
dop_err_list, cpf_err_list = [], []
pred_list, truth_list = [], []

for i in range(img_num):
        test_gt = test_data_add_reflect[i]

        real_gt = test_gt
        c, s, h, w = real_gt.shape
        gt_np = test_data_add_reflect[i]
        test_input = torch.tensor(test_gt).unsqueeze(0).cuda()

        # test_input /= opt.ratio
        # test_input[:, :, 0, :, :] / opt.ratio

        # input_meas = init_meas(test_gt, mask3d_batch_test, opt.input_setting)
        model.eval()

        filter_matrix_test = filter_matrix_test[:, :, :, :h, :w]

        with torch.no_grad():
            model_out = model(test_input, filter_matrix_test)
            model_out = model_out.view(1, 21, 4, h, w)
        out_tensor = model_out
        
        out_np = out_tensor[0].cpu().numpy()  # 形状: 21, 4, H, W
        gt_tensor = torch.tensor(test_data_add_reflect[i][np.newaxis, ...]).cuda()


        out_polar = out_np[:, 0, :, :] + out_np[:, 1, :, :]
        polar_img, _ = trans_multi_2_rgb(out_polar.transpose(1, 2, 0))
        gt_polar = gt_np[:, 0, :, :] + gt_np[:, 1, :, :]
        gt_polar_img, _ = trans_multi_2_rgb(gt_polar.transpose(1, 2, 0))
        # print('polar_img_shape:', out_polar.shape)
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(gt_polar_img)
        plt.axis('off')
        plt.title('GT Polarized')
        plt.subplot(1, 2, 2)
        plt.imshow(polar_img)   
        plt.axis('off')
        plt.title('Predicted Polarized')
        plt.tight_layout()
        plt.savefig(os.path.join(name, f'pred_polar_{i}.png'), bbox_inches='tight', pad_inches=0.1, dpi=200)
        plt.close()

        np.save(os.path.join(name, f'pred_{i}.npy'), out_np)
        np.save(os.path.join(name, f'gt_{i}.npy'), gt_tensor[0].cpu().numpy())

        # PSNR / SSIM
        psnr_val = torch_psnr(out_tensor[0, :, 0, :, :], gt_tensor[0, :, 0, :, :])
        ssim_val = torch_ssim(out_tensor[0, :, 0, :, :], gt_tensor[0, :, 0, :, :])
        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())

        # DoP / CPF
        # 重新组织维度顺序为: H, W, S, 4
        out_np_hw = np.transpose(out_np, (2, 3, 0, 1))  # H, W, 21, 4
        gt_np_hw = np.transpose(gt_np, (2, 3, 0, 1))    # H, W, 21, 4
        
        S0_p, S1_p, S2_p, S3_p = out_np_hw[..., 0], out_np_hw[..., 1], out_np_hw[..., 2], out_np_hw[..., 3]
        S0_g, S1_g, S2_g, S3_g = gt_np_hw[..., 0], gt_np_hw[..., 1], gt_np_hw[..., 2], gt_np_hw[..., 3]

        eps = 1e-8
        dop_pred = np.sqrt(S1_p**2 + S2_p**2 + S3_p**2) / (S0_p + eps)
        dop_gt = np.sqrt(S1_g**2 + S2_g**2 + S3_g**2) / (S0_g + eps)
        dop_pred = np.clip(dop_pred, 0, 1)
        dop_gt = np.clip(dop_gt, 0, 1)
        dop_err_list.append(np.mean((dop_pred - dop_gt)**2))

        cpf_pred = np.abs(S3_p) / (S0_p + eps)
        cpf_gt = np.abs(S3_g) / (S0_g + eps)
        cpf_pred = np.clip(cpf_pred, 0, 1)
        cpf_gt = np.clip(cpf_gt, 0, 1)
        cpf_err_list.append(np.mean((cpf_pred - cpf_gt)**2))

        # 保存预测和真值
        pred_list.append(out_np_hw)  # H x W x 21 x 4
        truth_list.append(gt_np_hw)  # H x W x 21 x 4

# 计算均值
psnr_mean = np.mean(psnr_list)
ssim_mean = np.mean(ssim_list)
dop_err_mean = np.mean(dop_err_list)
cpf_err_mean = np.mean(cpf_err_list)

pred_list = np.stack(pred_list, axis=0)
truth_list = np.stack(truth_list, axis=0)



draw_test_images(truth_list, pred_list, name, psnr_list, ssim_list)


