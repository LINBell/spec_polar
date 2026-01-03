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


model = MST_with_resnet(dim=21*4, stage=2, num_blocks=[2, 2, 2], need_noise=False, num_sp=5, need_mask_atten=False, raw_flag=True, resnet_layers=5).cuda()
# model = MST(dim=21*4, stage=2, num_blocks=[4, 7, 5], need_noise=opt.need_noise, num_sp=5, need_mask_atten=False, raw_flag=True).cuda()

model_path = '9_9/basic/model/model_epoch_265.pth'
model_path = '9_9/shuffle/model/model_epoch_171.pth'
model_path = '9_9/shuffle_full_size_real_denoise/model/model_epoch_255.pth'
model_path = 'ablation_test/cave_gen_specpolar_fullrandom/model/model_epoch_87.pth'
model_path = 'ablation_test/cave_gen_specpolar_fullrandom_v2/model/model_epoch_73.pth'
model_path = 'ablation_test2/mst_with_resnet_specpolar/model/model_epoch_103.pth'
model_path = 'ablation_test2/mst_with_resnet_specpolar/model/model_epoch_103.pth'
model_path = 'ablation_test2/mst_with_resnet_specpolar_v1/model/model_epoch_28.pth'
model_path = 'ablation_test2/mst_with_resnet_smooth_gendata/model/model_epoch_41.pth'
model_path = 'ablation_test2/mst_with_resnet_smooth_gendata_nocapdivde/model/model_epoch_106.pth'
# model_path = 'ablation_test2/mst_smooth_gendata/model/model_epoch_15.pth'
# model_path = '9_9/shuffle_new_norm/model/model_epoch_103.pth'
# model_path = '9_9/add_reflect_blur_random_angle/model/model_epoch_223.pth'
# model_path = '9_9/add_reflect/model/model_epoch_249.pth'
# model_path = '9_9/shuffle_full_size_real_denoise/model/model_epoch_255.pth'
# model_path = '9_9/add_reflect_denoise_random_angle/model/model_epoch_241.pth'
# model_path = '9_9/add_reflect_denoise/model/model_epoch_183.pth'

# model_path = '9_9/six_shuffle/model/model_epoch_212.pth'
# model_path = '9_9/shuffle_4/model/model_epoch_81.pth'
# model_path = '/data4T/lzj/mst_spectral/simulation/train_code/duizhao/zero/model/model_epoch_51.pth'

# model_path = '9_9/shuffle_full_size_real_denoise/model/model_epoch_255.pth'
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






# filter_matrix_test = filter_matrix_test[:, :, :, :220, :300]

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


test_data = np.load('color_raw/10x_3w_full.npy')  # 70 100 100
# test_data = np.load('color_raw/4x_2w_full.npy')  # 70 100 100
# test_data = np.load('color_raw/LR_2w.npy')  # 70 100 100
# test_data = np.load('color_raw/onlyL_3w.npy')  # 70 100 100

# test_data = np.load('color_raw/4x_2w_full.npy')  # 70 100 100
test_data = np.load('color_raw/white_80_100.npy')
test_data = np.load('color_raw/color_3w_280_400.npy')
test_data = np.load('color_raw/10.10_color_2w5.npy')
test_data = np.load('color_raw/11.2_origin.npy')
test_data = np.load('color_raw/11.2_led.npy')
test_data = np.load('color_raw/11.2_led_wi0.npy')
test_data = np.load('color_raw/11.2_led_wi90.npy')
test_data = np.load('color_raw/11.2_led2.npy')
test_data = np.load('color_raw/11.8_small_color.npy')
test_data = np.load('color_raw/11.8_small_color4.npy')


test_data = np.load('color_raw_new/11.8_small_color.npy')
# test_data = np.load('color_raw_new/reflect_raw.npy')
# test_data = np.load('color_raw_new/reflect.npy')
# test_data = np.load('color_raw_new/reflect2.npy')
# test_data = np.load('color_raw_new/reflect2_wi_polar.npy')
# test_data = np.load('color_raw_new/reflect3.npy')

# test_data = np.load('color_raw/10.10_color_2w5.npy')


# test_data = np.load('color_raw/10.10_thorlabs_2w5.npy')
# test_data = np.load('color_raw/4x_2w_real_4.npy')
# test_data = np.load('color_raw/real_10.npy')
# test_data = np.load('color_raw/real_4.npy')
# test_data = np.load('color_raw/reflect3_0.4w.npy')
# test_data = np.load('color_raw/10.17_reflect_2w.npy')
# test_data = np.load('color_raw/10_18_8.npy')

test_data = np.load('color_raw_new/11.20/H.npy')
test_data = np.load('color_raw_new/11.20/first_shot.npy')
# test_data = np.load('color_raw_new/12.7/15.npy')
# test_data = np.load('color_raw_new/12.7/12.npy')
# test_data = np.load('color_raw_new/first_shot.npy')

# test_data = np.load('color_raw/11.8_small_color.npy')  # 70 100 100test_data = np.load('color_raw/11.8_small_color.npy')  # 70 100 100
# test_data = np.load('color_raw/10.10_color_2w5.npy')  # 70 100 100


print(test_data.shape)
h, w = test_data.shape[0], test_data.shape[1]
filter_matrix_test = filter_matrix_test[:, :, :, :h, :w]


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
    model_out = model_out.view(batch_size, 21, 4, h, w).squeeze(0)
    # model_out = model_out.view(batch_size, 21, 4, 220, 300).squeeze(0)
    model_out = model_out.detach().cpu().numpy()



print(f'原始形状: {model_out.shape}')




def trans_six_to_four(img, norm=False):
    c, s, h, w = img.shape
    four = np.zeros((c, 4, h, w))
    

    H = img[:, 0, :, :]
    V = img[:, 1, :, :]
    A = img[:, 2, :, :]
    D = img[:, 3, :, :]
    L = img[:, 4, :, :]
    R = img[:, 5, :, :]
    
    four[:, 0, :, :] = H + V
    four[:, 1, :, :] = H - V
    four[:, 2, :, :] = A - D
    four[:, 3, :, :]= L - R


    
    return four

# model_out = trans_six_to_four(model_out)



save_path = f'raw_temp'
os.makedirs(save_path, exist_ok=True)
np.save(os.path.join(save_path, 'raw_full.npy'), model_out)
channel_index = [0, 10, 20]

# rgb_st, rgb = trans_multi_2_rgb(model_out[:, 0, :, :].transpose(1, 2, 0))
rgb_v2 = trans_multi_2_rgb_v2(model_out[:, 0, :, :].transpose(1, 2, 0))
rgbs, rgb = trans_multi_2_rgb(model_out[:, 0, :, :].transpose(1, 2, 0))



rgbs_0, rgb_0 = trans_multi_2_rgb((model_out[:, 0, :, :]+model_out[:, 1, :, :]).transpose(1, 2, 0))
# np.save(os.path.join(save_path, 'raw.npy'), final_image_full[channel_index, 0, :, :].transpose(1, 2, 0))

np.save(os.path.join(save_path, 'raw.npy'), rgb)
plt.figure()
# plt.imshow(final_image_full[0, 0, :, :].transpose(1, 2, 0), cmap='gray')
plt.subplot(1, 3, 1)
plt.imshow(rgbs)
plt.axis('off')
plt.subplot(1, 3, 2)
# plt.imshow(rgb_st)
plt.imshow(rgb_v2)
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(rgbs_0)
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








