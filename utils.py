import copy
import math
from matplotlib import figure, pyplot as plt
import scipy.io as sio
import os
import numpy as np
import torch
import logging
import random
# from ssim_torch import ssim
import cv2
from pytorch_msssim import ssim

reflect_led_spectrum = [0.66228936, 0.63492155, 0.4348925,  0.29973852, 0.26862289, 0.35252759,
 0.49686228, 0.6696688,  0.80569436, 0.86836141, 0.91586287, 0.95342823,
 0.97024985, 1.         , 0.94590353, 0.9277455,  0.85031958, 0.77135386,
 0.68201627, 0.61246368, 0.53503776]
reflect_led_spectrum = np.array(reflect_led_spectrum)

def draw_test_images_six(truth, pred, path, psnr_list, ssim_list):  # bs h w c s
    channle_index = [0, 10, 20]
    if not os.path.exists(path):
        os.makedirs(path)
    bs, h, w, c, s = truth.shape
    # print(truth.shape)
    for index in range(bs):
        if index == 1:
            np.save(os.path.join(path, 'gt.npy'), truth[index, :, :, :, 0] + truth[index, :, :, :, 1] )
            np.save(os.path.join(path, 'out.npy'), pred[index, :, :, :, 0] + pred[index, :, :, :, 1] )
        # 取第10个波段的S0和S1，求平均
        gt = truth[index, :, :, :, 0] + truth[index, :, :, :, 1] 
        out = pred[index, :, :, :, 0] + pred[index, :, :, :, 1] 
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


        for i in range(6):
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



# def draw_test_images(truth, pred, path, psnr_list, ssim_list):  # bs h w c s
#     channle_index = [0, 10, 20]
#     if not os.path.exists(path):
#         os.makedirs(path)
#     bs, h, w, c, s = truth.shape
#     # print(truth.shape)
#     for index in range(bs):
#         if index == 1:
#             np.save(os.path.join(path, 'gt.npy'), truth[index, :, :, :, 0] )
#             np.save(os.path.join(path, 'out.npy'), pred[index, :, :, :, 0] )
#         # 取第10个波段的S0和S1，求平均
#         if truth.shape[-1] == 6:
#             gt = trans_six_to_four(truth[index])[:, :, :, 0]
#             out = trans_six_to_four(pred[index])[:, :, :, 0]
#         else:
#             gt = truth[index, :, :, :, 0] 
#             out = pred[index, :, :, :, 0] 
#         # print(f"gt shape after indexing: {gt.shape}")
#         # print(f"out shape after indexing: {out.shape}")
#         plt.figure(figsize=(8, 4))
#         plt.subplot(1, 2, 1)
#         # plt.imshow(gt.transpose(1, 2, 0))
#         gt_disp, _ = trans_multi_2_rgb(gt)
#         plt.imshow(gt_disp)
#         plt.title('GT')
#         plt.axis('off')

#         plt.subplot(1, 2, 2)
#         # plt.imshow(out.transpose(1, 2, 0))
#         out_disp, _ = trans_multi_2_rgb(out)
#         plt.imshow(out_disp)
#         plt.title('out')
#         plt.axis('off')
#         save_path = os.path.join(path, "{}_{:.2f}_{:.2f}.png".format(index, psnr_list[index], ssim_list[index]))
#         plt.tight_layout()
#         plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=200)
#         plt.close()




#         for i in range(4):
#             select_x = random.randint(0, h-1)
#             select_y = random.randint(0, w-1)
#             # 提取该像素点所有波段的光谱（S0和S1平均）
#             gt_spec = truth[index, select_x, select_y, :, i] 
#             out_spec = pred[index, select_x, select_y, :, i] 
#             plt.figure()
#             plt.plot(gt_spec, 'r', label='GT')
#             plt.plot(out_spec, 'y', label='out')
#             plt.title(f'Pixel ({select_x},{select_y},States{i}) Spectrum')
#             plt.xlabel('Wavelength Index')
#             plt.ylabel('Intensity')
#             plt.legend()
#             plt.tight_layout()
#             plt.savefig(os.path.join(path, f"{index}_spec_{i}.png"), bbox_inches='tight', pad_inches=0.1, dpi=200)
#             plt.close()


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
        if truth.shape[-1] == 6:
            gt = trans_six_to_four(truth[index])[:, :, :, 0]
            out = trans_six_to_four(pred[index])[:, :, :, 0]
        else:
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

        # 绘制S1、S2、S3的灰度图子图
        plt.figure(figsize=(12, 8))
        for i in range(3):
            # S1, S2, S3 分别对应 i=0,1,2
            plt.subplot(2, 3, i+1)  # 第一行：GT
            gt_gray = truth[index, :, :, 10, i+1]  # 取第一个通道的S1/S2/S3
            plt.imshow(gt_gray, cmap='gray')
            plt.title(f'GT S{i+1}')
            plt.axis('off')
            
            plt.subplot(2, 3, i+4)  # 第二行：Pred
            pred_gray = pred[index, :, :, 10, i+1]  # 取第一个通道的S1/S2/S3
            plt.imshow(pred_gray, cmap='gray')
            plt.title(f'Pred S{i+1}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(path, f"{index}_S1S2S3_gray.png"), bbox_inches='tight', pad_inches=0.1, dpi=200)
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

def draw_test_images_only_polar(truth, pred, path, psnr_list, ssim_list):  # bs h w c s

    if not os.path.exists(path):
        os.makedirs(path)
    bs, h, w, s = truth.shape
    print(truth.shape)
    for index in range(bs):
        if index == 1:
            np.save(os.path.join(path, 'gt.npy'), truth[index, :, :, 0] )
            np.save(os.path.join(path, 'out.npy'), pred[index, :, :, 0] )


        # 设置更大的图形尺寸
        plt.figure(figsize=(16, 8))  # 宽度16英寸，高度8英寸

        # 定义标题
        gt_titles = ['GT S0', 'GT S1', 'GT S2', 'GT S3']
        pred_titles = ['Pred S0', 'Pred S1', 'Pred S2', 'Pred S3']

        # 绘制GT图像
        for i in range(4):
            plt.subplot(2, 4, i+1)
            plt.imshow(truth[index, :, :, i], cmap='gray')
            plt.axis('off')
            plt.title(gt_titles[i], fontsize=12)

        # 绘制Pred图像
        for i in range(4):
            plt.subplot(2, 4, i+5)
            plt.imshow(pred[index, :, :, i], cmap='gray')
            plt.axis('off')
            plt.title(pred_titles[i], fontsize=12)

        # 调整子图间距
        plt.subplots_adjust(wspace=0.1, hspace=0.15)  # 减少子图间距

        # 保存图像
        plt.savefig(
            os.path.join(path, f"{index}_{psnr_list[index]:.2f}_{ssim_list[index]:.2f}.png"),
            dpi=150,  # 提高分辨率
            bbox_inches='tight',  # 紧凑布局
            pad_inches=0.1  # 边缘留白
        )
        plt.close()  # 关闭图形以释放内存

        
def draw_test_images_only_spec(truth, pred, path, psnr_list, ssim_list):  # bs h w c s

    if not os.path.exists(path):
        os.makedirs(path)
    bs, h, w, c = truth.shape
    # print(truth.shape)
    for index in range(bs):
        if index == 1:
            np.save(os.path.join(path, 'gt.npy'), truth[index, :, :, :] )
            np.save(os.path.join(path, 'out.npy'), pred[index, :, :, :] )

        gt = truth[index, :, :, :]
        out = pred[index, :, :, :]
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
            gt_spec = truth[index, select_x, select_y, :] 
            out_spec = pred[index, select_x, select_y, :] 
            plt.figure()
            plt.plot(gt_spec, 'r', label='GT')
            plt.plot(out_spec, 'y', label='out')
            plt.title(f'Pixel ({select_x},{select_y},Times{i}) Spectrum')
            plt.xlabel('Wavelength Index')
            plt.ylabel('Intensity')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(path, f"{index}_spec_{i}.png"), bbox_inches='tight', pad_inches=0.1, dpi=200)



def draw_test_images_single_dimention(truth, pred, path, psnr_list, ssim_list, dimention=None):  # bs h w c s
    channle_index = [0, 10, 20]
    if not os.path.exists(path):
        os.makedirs(path)
    bs, h, w, c = truth.shape
    # print(truth.shape)
    for index in range(bs):
        if index == 1:
            np.save(os.path.join(path, 'gt.npy'), truth[index, :, :, 0] )
            np.save(os.path.join(path, 'out.npy'), pred[index, :, :, 0] )

        gt = truth[index, :, :, 0] 
        out = pred[index, :, :, 0] 
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

        # 绘制S1、S2、S3的灰度图子图
        plt.figure(figsize=(12, 8))
        for i in range(3):
            # S1, S2, S3 分别对应 i=0,1,2
            plt.subplot(2, 3, i+1)  # 第一行：GT
            gt_gray = truth[index, :, :, 10, i+1]  # 取第一个通道的S1/S2/S3
            plt.imshow(gt_gray, cmap='gray')
            plt.title(f'GT S{i+1}')
            plt.axis('off')
            
            plt.subplot(2, 3, i+4)  # 第二行：Pred
            pred_gray = pred[index, :, :, 10, i+1]  # 取第一个通道的S1/S2/S3
            plt.imshow(pred_gray, cmap='gray')
            plt.title(f'Pred S{i+1}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(path, f"{index}_S1S2S3_gray.png"), bbox_inches='tight', pad_inches=0.1, dpi=200)
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

def process_data(cap_data, num_sp):  # list 9 x 5 1 80 80   need 5 1 240 240
    bs, c, h, w = cap_data[0].shape
    full_img = torch.zeros([bs, 1, h * num_sp, w * num_sp]).cuda()
    index_x = torch.arange(0, h * num_sp, num_sp)
    index_y = torch.arange(0, w * num_sp, num_sp)
    for num in range(num_sp * num_sp):
        x = num // num_sp
        y = num % num_sp

        # data = cap_data[num].cpu().detach().numpy()  # 5 28 100 100
        data = cap_data[num]

        full_img[:, :, y + index_x[:, None], x + index_y] = data

    return full_img

sigma = 3
x = np.linspace(1, 21, 21)
real_center = list(range(1,22))

def getG(sigma, index):
    para = (2 * math.pi) ** 0.5 * sigma
    zhishu = -(x - real_center[index]) ** 2 / (2 * sigma ** 2)

    y = [math.exp(x) / para for x in zhishu]
    return y


num_theta = 21

def cal_fisher(input, block):  # [1, 35]
    input0 = input  # 15 21 120 120

    # input0 = torch.Tensor(input0).reshape(1, 35, 1, 1).cuda()
    s0, _, _, _, _ = block.capture(input0)  # cap_data 15 25 60 60
    bs, c, h, w = s0.shape
    derivatives_pattern = torch.zeros([bs, 25, h, w, 21]).cuda()
    for i in range(num_theta):
        input1 = copy.deepcopy(input).cuda()
        delt = np.array([x / 1000.0 for x in getG(0.01, i)]).reshape(1, 21, 1, 1)  # 0.03989
        delt = torch.Tensor(delt).cuda()
        input1 = input1 + delt
        input1 = torch.Tensor(input1).cuda().float()
        s1, _, _, _, _ = block.capture(input1)  # 15 9 40 40

        derivatives = (s1 - s0) / 0.03989
        derivatives_pattern[:, :, :, :, i] = derivatives  # 15 9 40 40 21

    return derivatives_pattern


def cal_fisher_full(input, block):  # [1, 35]
    input0 = input  # 15 21 120 120
    # input0 = torch.Tensor(input0).reshape(1, 35, 1, 1).cuda()
    _, x_p0, _, _, _ = block.compress(input0)  # cap_data 15 25 60 60   --- 8 1 160 160
    x_p0 = x_p0.permute(0, 3, 1, 2)
    bs, c, h, w = x_p0.shape
    derivatives_pattern = torch.zeros([bs, 1, h, w, 21]).cuda()
    for i in range(num_theta):
        input1 = copy.deepcopy(input).cuda()
        delt = np.array([x / 1000.0 for x in getG(0.01, i)]).reshape(1, 21, 1, 1)  # 0.03989
        delt = torch.Tensor(delt).cuda()
        input1 = input1 + delt
        input1 = torch.Tensor(input1).cuda().float()
        _, x_p1, _, _, _ = block.compress(input1)  # 15 9 40 40
        x_p1 = x_p1.permute(0, 3, 1, 2)

        derivatives = (x_p1 - x_p0) / 0.03989
        derivatives_pattern[:, :, :, :, i] = derivatives  # 15 9 40 40 21

    return derivatives_pattern





def generate_masks(mask_path, batch_size):
    # mask = sio.loadmat(mask_path + '/mask.mat')
    mask_path = os.path.join(mask_path, 'mask.mat')



    if mask_path is None:
        raise ValueError("mask_path is None. Please check the --mask_path argument.")

    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found at {mask_path}")

    mask = sio.loadmat(mask_path)
    mask = mask['mask']
    mask3d = np.tile(mask[:, :, np.newaxis], (1, 1, 28))
    mask3d = np.transpose(mask3d, [2, 0, 1])
    mask3d = torch.from_numpy(mask3d)
    [nC, H, W] = mask3d.shape
    mask3d_batch = mask3d.expand([batch_size, nC, H, W]).cuda().float()
    return mask3d_batch

def generate_shift_masks(mask_path, batch_size):
    mask = sio.loadmat(mask_path + '/mask_3d_shift.mat')
    mask_3d_shift = mask['mask_3d_shift']
    mask_3d_shift = np.transpose(mask_3d_shift, [2, 0, 1])
    mask_3d_shift = torch.from_numpy(mask_3d_shift)
    [nC, H, W] = mask_3d_shift.shape
    Phi_batch = mask_3d_shift.expand([batch_size, nC, H, W]).cuda().float()
    Phi_s_batch = torch.sum(Phi_batch**2,1)
    Phi_s_batch[Phi_s_batch==0] = 1
    # print(Phi_batch.shape, Phi_s_batch.shape)
    return Phi_batch, Phi_s_batch

def LoadTraining_npy(path, select_num=None):
    imgs = []
    # 获取并排序场景列表
    scene_list = sorted([f for f in os.listdir(path) if f.endswith('.npy')])
    
    print('Total .npy scenes found:', len(scene_list))
    
    # 如果指定了select_num，限制加载数量
    if select_num is not None:
        scene_list = scene_list[:select_num]
        print(f'Will load first {select_num} scenes')

    for i in range(len(scene_list)):
        scene_path = os.path.join(path, scene_list[i])

        # 检查文件是否是 .npy 文件
        if not scene_list[i].endswith('.npy'):
            continue

        # 加载 .npy 文件
        img = np.load(scene_path)

        # 如果需要归一化或其他处理，可以在这里添加
        img = img.astype(np.float32)  # 转换为 float32
        img = (img - np.min(img)) / (np.max(img) - np.min(img))

        imgs.append(img)
        print('Scene {} is loaded. {}'.format(i, scene_list[i]))

    return imgs


def LoadTraining(path, select_num=None):
    imgs = []
    scene_list = os.listdir(path)
    scene_list.sort()
    print('training sences:', len(scene_list))
    if select_num is not None:
        scene_list = scene_list[:select_num]
        print(f'Will load first {select_num} scenes')
    for i in range(len(scene_list)):
    # for i in range(5):
        scene_path = path + '/' + scene_list[i]
        # print(scene_path)
        scene_num = int(scene_list[i].split('.')[0][5:])
        if scene_num<=205:
            if 'mat' not in scene_path:
                continue
            img_dict = sio.loadmat(scene_path)
            if "img_expand" in img_dict:
                img = img_dict['img_expand'] / 65536.
            elif "img" in img_dict:
                img = img_dict['img'] / 65536.
            img = img.astype(np.float32)
            
            imgs.append(img[:, :, [0,2,4,6,8,9,11,13,14,16,17,18,19,20,21,22,23,24,26,26,27], np.newaxis])
            print('Sence {} is loaded. {}'.format(i, scene_list[i]))
    return imgs

def trans_specpolar_to_polar_cave(img):
    h, w, c= img.shape
    select_index = random.sample(range(c), 7)
    s0 = img[:, :, select_index[0]]
    s1 = img[:, :, select_index[1]] - img[:, :, select_index[2]]
    s2 = img[:, :, select_index[3]] - img[:, :, select_index[4]]
    s3 = img[:, :, select_index[5]] - img[:, :, select_index[6]]
    polar = np.zeros((h, w, 4))
    polar[:, :, 0] = s0
    polar[:, :, 1] = s1
    polar[:, :, 2] = s2
    polar[:, :, 3] = s3
    return polar[..., np.newaxis].transpose(0, 1, 3, 2) # h w c s 

def LoadTraining_gen_specpolar(path, select_num=None):
    imgs = []
    scene_list = os.listdir(path)
    scene_list.sort()
    print('training sences:', len(scene_list))
    if select_num is not None:
        scene_list = scene_list[:select_num]
        print(f'Will load first {select_num} scenes')
    for i in range(len(scene_list)):
    # for i in range(5):
        scene_path = path + '/' + scene_list[i]
        # print(scene_path)
        scene_num = int(scene_list[i].split('.')[0][5:])
        if scene_num<=205:
            if 'mat' not in scene_path:
                continue
            img_dict = sio.loadmat(scene_path)
            if "img_expand" in img_dict:
                img = img_dict['img_expand'] / 65536.
            elif "img" in img_dict:
                img = img_dict['img'] / 65536.
            img = img.astype(np.float32)
            imgs.append(construct_stokes_random_combinations(img).transpose(2, 3, 0, 1))
            print('Sence {} is loaded. {}'.format(i, scene_list[i]))
    return imgs


def LoadTraining_gen_polar(path, select_num=None):
    imgs = []
    scene_list = os.listdir(path)
    scene_list.sort()
    print('training sences:', len(scene_list))
    if select_num is not None:
        scene_list = scene_list[:select_num]
        print(f'Will load first {select_num} scenes')
    for i in range(len(scene_list)):
    # for i in range(5):
        scene_path = path + '/' + scene_list[i]
        # print(scene_path)
        scene_num = int(scene_list[i].split('.')[0][5:])
        if scene_num<=205:
            if 'mat' not in scene_path:
                continue
            img_dict = sio.loadmat(scene_path)
            if "img_expand" in img_dict:
                img = img_dict['img_expand'] / 65536.
            elif "img" in img_dict:
                img = img_dict['img'] / 65536.
            img = img.astype(np.float32)
            img = img[:, :, [0,2,4,6,8,9,11,13,14,16,17,18,19,20,21,22,23,24,25,26,27]]
            
            imgs.append(trans_specpolar_to_polar_cave(img))
            print('Sence {} is loaded. {}'.format(i, scene_list[i]))
    return imgs


# def construct_stokes_random_combinations(data, num_groups=21):
#     """
#     生成21组随机组合的全斯托克斯数据
#     """

    
#     H, W, C = data.shape
#     stokes_list = []
    
#     for _ in range(num_groups):
#         # 随机选择7个不重复的通道索引
#         indices = np.random.choice(28, 7, replace=False)
#         # indices.sort()  # 可选：保持顺序
        
#         selected_channels = data[..., indices]  # (H, W, 7)
        
#         # 构造斯托克斯参数
#         S0 = selected_channels[..., 0]
#         S1 = selected_channels[..., 1] - selected_channels[..., 2]
#         S2 = selected_channels[..., 3] - selected_channels[..., 4]
#         S3 = selected_channels[..., 5] - selected_channels[..., 6]
        
#         stokes = np.stack([S0, S1, S2, S3], axis=-1)
#         stokes_list.append(stokes.transpose(2, 0, 1))
    
#     return np.array(stokes_list)

# def construct_stokes_random_combinations(data, num_groups=None):  """cave_gen_specpolar_fullrandom  到了180多epoch只保存到80"""
#     """
#     使用滑动窗口生成全斯托克斯数据
#     窗口大小固定为7，连续滑动覆盖28个通道
#     默认生成22个窗口（28-7+1=22），但可以指定数量
#     """
#     H, W, C = data.shape  # C=28
    
#     # 计算最大可能的窗口数
#     max_windows = C - 7   # 28-7+1 = 22
    
#     # 如果没有指定数量，使用最大值
#     if num_groups is None:
#         num_groups = max_windows
#     else:
#         # 确保不超过最大值
#         num_groups = min(num_groups, max_windows)
    
#     stokes_list = []
    
#     # 生成均匀分布的起始点，确保覆盖整个光谱范围
#     if num_groups == max_windows:
#         # 如果是最大数量，使用所有可能的窗口
#         start_indices = range(max_windows)
#     else:
#         # 如果不是最大数量，均匀选择起始点
#         step = max(1, max_windows // num_groups)
#         start_indices = [i * step for i in range(num_groups)]
#         # 确保最后一个窗口不会越界
#         start_indices = [min(s, max_windows-1) for s in start_indices]
    
#     for start_idx in start_indices:
#         # 选择连续的7个通道
#         window_data = data[..., start_idx:start_idx+7]  # (H, W, 7)
        
#         # 在窗口内随机选择I1-I6（不重复）
#         # 创建0-6的随机排列
#         perm = np.random.permutation(7)
        
#         # 确保I0是窗口内7个通道的平均值
#         I0 = np.mean(window_data, axis=-1)  # (H, W)
        
#         # 随机选择I1-I6
#         I1 = window_data[..., perm[0]]
#         I2 = window_data[..., perm[1]]
#         I3 = window_data[..., perm[2]]
#         I4 = window_data[..., perm[3]]
#         I5 = window_data[..., perm[4]]
#         I6 = window_data[..., perm[5]]
        
#         # 构造斯托克斯参数
#         S0 = I0
#         S1 = I1 - I2
#         S2 = I3 - I4
#         S3 = I5 - I6
        
#         stokes = np.stack([S0, S1, S2, S3], axis=-1)  # (H, W, 4)
#         stokes_list.append(stokes.transpose(2, 0, 1))  # (4, H, W)
    
#     return np.array(stokes_list)  # (num_groups, 4, H, W)

def construct_stokes_random_combinations(data):  
    """
    cave_gen_specpolar_fullrandomv2  
    从28个通道中固定挑选21个通道作为S0
    对于每个S0，随机从28个通道中挑选I1-I6生成斯托克斯矢量
    """
    H, W, C = data.shape  # C=28
    
    # 你指定的21个通道索引（注意26重复了，我调整了一下）
    s0_indices = [0, 2, 4, 6, 8, 9, 11, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
    
    # 确保索引在有效范围内
    s0_indices = [idx for idx in s0_indices if idx < C]
    
    if len(s0_indices) != 21:
        print(f"警告：指定的S0通道数为{len(s0_indices)}，不是21个")
    
    stokes_list = []
    
    for s0_idx in s0_indices:
        # S0: 直接使用指定的通道
        S0 = data[..., s0_idx]  # (H, W)
        
        # 从28个通道中随机选择6个不同的通道作为I1-I6
        # 确保不包括当前S0的通道，避免自相关
        all_channels = list(range(C))
        all_channels.remove(s0_idx)  # 移除S0通道
        
        # 随机选择6个不同的通道
        selected_indices = np.random.choice(all_channels, size=6, replace=False)
        
        # 获取对应的通道数据
        I1 = data[..., selected_indices[0]]
        I2 = data[..., selected_indices[1]]
        I3 = data[..., selected_indices[2]]
        I4 = data[..., selected_indices[3]]
        I5 = data[..., selected_indices[4]]
        I6 = data[..., selected_indices[5]]
        
        # 构造斯托克斯参数
        # S0 = I0 (已指定)
        S1 = I1 - I2
        S2 = I3 - I4
        S3 = I5 - I6
        
        stokes = np.stack([S0, S1, S2, S3], axis=-1)  # (H, W, 4)
        stokes_list.append(stokes.transpose(2, 0, 1))  # (4, H, W)
    
    return np.array(stokes_list)  # (21, 4, H, W)


# def construct_stokes_random_combinations(data, window_size=3):
#     """
#     使用滑动窗口平均来平滑通道选择，减少突变
#     """
#     H, W, C = data.shape
    
#     # 固定S0：使用多个通道的平均值
#     s0_indices = [0, 2, 4, 6, 8, 9, 11, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
    
#     stokes_list = []
    
#     for s0_idx in s0_indices:
#         # S0使用窗口平均，而不是单个通道
#         start = max(0, s0_idx - window_size // 2)
#         end = min(C, s0_idx + window_size // 2 + 1)
#         S0 = np.mean(data[..., start:end], axis=-1)
        
#         # 从邻近波长中选择通道，而不是完全随机
#         # 确保选择的通道在波长上是邻近的
#         center_channel = s0_idx
#         candidate_indices = list(range(max(0, center_channel-5), min(C, center_channel+6)))
#         candidate_indices.remove(center_channel)
        
#         # 按波长顺序选择6个通道
#         if len(candidate_indices) >= 6:
#             # 选择等间距的6个通道，保持波长连续性
#             step = len(candidate_indices) // 6
#             selected_indices = [candidate_indices[i*step] for i in range(6)]
#         else:
#             selected_indices = np.random.choice(candidate_indices, size=6, replace=True)
        
#         # 获取数据并进行轻微平滑
#         selected_data = []
#         for idx in selected_indices:
#             # 对每个选择的通道进行邻域平均
#             start_smooth = max(0, idx - 1)
#             end_smooth = min(C, idx + 2)
#             channel_data = np.mean(data[..., start_smooth:end_smooth], axis=-1)
#             selected_data.append(channel_data)
        
#         # 构造斯托克斯参数
#         S1 = selected_data[0] - selected_data[1]
#         S2 = selected_data[2] - selected_data[3]
#         S3 = selected_data[4] - selected_data[5]
        
#         stokes = np.stack([S0, S1, S2, S3], axis=-1)
#         stokes_list.append(stokes.transpose(2, 0, 1))
    
#     return np.array(stokes_list)



def LoadTest_gen_polar(path_test):
    scene_list = os.listdir(path_test)
    scene_list.sort()
    test_data = np.zeros((len(scene_list), 256, 256, 1, 4))
    for i in range(len(scene_list)):
        scene_path = path_test + '/' +  scene_list[i]
        img = sio.loadmat(scene_path)['img']
        img = img[:, :, [0,2,4,6,8,9,11,13,14,16,17,18,19,20,21,22,23,24,26,26,27]]
        img = trans_specpolar_to_polar_cave(img) # hwcs
        img = img.astype(np.float32)
        test_data[i, :, :, :, :] = img
    test_data = torch.from_numpy(test_data)
    return test_data

def LoadTest_gen_specpolar(path_test):
    scene_list = os.listdir(path_test)
    scene_list.sort()
    test_data = np.zeros((len(scene_list), 21, 4, 256, 256))
    for i in range(len(scene_list)):
        scene_path = path_test + '/' +  scene_list[i]
        img = sio.loadmat(scene_path)['img']

        img = construct_stokes_random_combinations(img) # hwcs
        img = img.astype(np.float32)
        test_data[i, :, :, :, :] = img
    test_data = torch.from_numpy(test_data).float()
    return test_data


def LoadTest(path_test):
    scene_list = os.listdir(path_test)
    scene_list.sort()
    test_data = np.zeros((len(scene_list), 256, 256, 28))
    for i in range(len(scene_list)):
        scene_path = path_test + '/' +  scene_list[i]
        img = sio.loadmat(scene_path)['img']
        test_data[i, :, :, :] = img
    test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2))).float()
    return test_data[:, [0,2,4,6,8,9,11,13,14,16,17,18,19,20,21,22,23,24,26,26,27], np.newaxis, :, :]

def LoadMeasurement(path_test_meas):
    img = sio.loadmat(path_test_meas)['simulation_test']
    test_data = img
    test_data = torch.from_numpy(test_data)
    return test_data

# We find that this calculation method is more close to DGSMP's.
def torch_psnr(img, ref):  # input [28,256,256]
    img = (img*256).round()
    ref = (ref*256).round()
    nC = img.shape[0]
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        psnr += 10 * torch.log10((255*255)/mse)
    return psnr / nC

# def torch_ssim(img, ref):  # input [28,256,256]
#     return ssim(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0))

def torch_ssim(img, ref):  # input [28,256,256]
    # pytorch-msssim 的 ssim 函数期望输入为 [B, C, H, W]
    # 并且需要指定 data_range 参数
    img_unsqueezed = torch.unsqueeze(img, 0)  # [1, 28, 256, 256]
    ref_unsqueezed = torch.unsqueeze(ref, 0)  # [1, 28, 256, 256]
    return ssim(img_unsqueezed, ref_unsqueezed, data_range=1.0, size_average=True)

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

def shuffle_crop(train_data, batch_size, crop_size=160, argument=True):
    if argument:
        gt_batch = []
        # The first half data use the original data.
        index = np.random.choice(range(len(train_data)), batch_size//2)
        processed_data = np.zeros((batch_size//2, crop_size, crop_size, 21), dtype=np.float32)
        for i in range(batch_size//2):
            img = train_data[index[i]]
            h, w, _ = img.shape
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            processed_data[i, :, :, :] = img[x_index:x_index + crop_size, y_index:y_index + crop_size, :]
        processed_data = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2))).cuda().float()
        for i in range(processed_data.shape[0]):
            gt_batch.append(arguement_1(processed_data[i]))

        # The other half data use splicing.
        processed_data = np.zeros((4, crop_size//2, crop_size//2, 21), dtype=np.float32)
        for i in range(batch_size - batch_size // 2):
            sample_list = np.random.randint(0, len(train_data), 4)
            for j in range(4):
                x_index = np.random.randint(0, h-crop_size//2)
                y_index = np.random.randint(0, w-crop_size//2)
                processed_data[j] = train_data[sample_list[j]][x_index:x_index+crop_size//2,y_index:y_index+crop_size//2,:]
            gt_batch_2 = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2))).cuda()  # [4,28,128,128]
            gt_batch.append(arguement_2(gt_batch_2, crop_size))
        gt_batch = torch.stack(gt_batch, dim=0)
        return gt_batch
    else:
        index = np.random.choice(range(len(train_data)), batch_size)
        processed_data = np.zeros((batch_size, crop_size, crop_size, 21), dtype=np.float32)
        for i in range(batch_size):
            h, w, _ = train_data[index[i]].shape
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            processed_data[i, :, :, :] = train_data[index[i]][x_index:x_index + crop_size, y_index:y_index + crop_size, :]
        gt_batch = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2)))
        return gt_batch

def arguement_1(x):
    """
    :param x: c,h,w
    :return: c,h,w
    """
    rotTimes = random.randint(0, 3)
    vFlip = random.randint(0, 1)
    hFlip = random.randint(0, 1)
    # Random rotation
    for j in range(rotTimes):
        x = torch.rot90(x, dims=(1, 2))
    # Random vertical Flip
    for j in range(vFlip):
        x = torch.flip(x, dims=(2,))
    # Random horizontal Flip
    for j in range(hFlip):
        x = torch.flip(x, dims=(1,))
    return x

def arguement_2(generate_gt, crop_size):
    c, h, w = generate_gt.shape[1],crop_size,crop_size
    divid_point_h = crop_size//2
    divid_point_w = crop_size//2
    output_img = torch.zeros(c,h,w).cuda()
    output_img[:, :divid_point_h, :divid_point_w] = generate_gt[0]
    output_img[:, :divid_point_h, divid_point_w:] = generate_gt[1]
    output_img[:, divid_point_h:, :divid_point_w] = generate_gt[2]
    output_img[:, divid_point_h:, divid_point_w:] = generate_gt[3]
    return output_img


# def normalize_stokes(data):
#     """
#     归一化高光谱-偏振数据
#     输入数据形状: [h, w, 21, 4]
#     归一化规则:
#         s0 (第4维的第0个元素) -> [0, 1]
#         s1, s2, s3 (第4维的第1-3个元素) -> [-1, 1]
#     """
#     # 创建数据的副本以避免修改原始数据
#     normalized_data = data.copy()

#     # 归一化s0到[0,1]范围
#     s0 = data[..., 0]  # 提取s0分量
#     s0_min = np.min(s0)
#     s0_max = np.max(s0)
#     normalized_data[..., 0] = (s0 - s0_min) / (s0_max - s0_min)

#     # 归一化s1,s2,s3到[-1,1]范围
#     for i in range(1, 4):  # 遍历s1,s2,s3
#         si = data[..., i]  # 提取当前分量
#         si_max = np.max(np.abs(si))  # 找到最大绝对值
#         if si_max > 0:  # 避免除以0
#             normalized_data[..., i] = si / si_max

#     return normalized_data


def normalize_stokes(data):
    """简单但安全的归一化"""
    normalized_data = data.copy()
    
    # 1. 先修正数据（强制满足物理约束）
    s0 = data[..., 0]
    s1 = data[..., 1]
    s2 = data[..., 2]
    s3 = data[..., 3]
    
    # 确保s0非负
    s0 = np.maximum(s0, 0)
    
    # 计算当前偏振度
    sum_sq = s1**2 + s2**2 + s3**2
    dop = np.sqrt(np.maximum(sum_sq, 0)) / (s0 + 1e-10)
    
    # 缩放使偏振度不超过1
    scale = np.maximum(dop, 1.0)
    s1 = s1 / scale
    s2 = s2 / scale
    s3 = s3 / scale
    
    # 2. 归一化
    # s0归一化到[0,1]
    s0_min = np.min(s0)
    s0_max = np.max(s0)
    s0_norm = (s0 - s0_min) / (s0_max - s0_min + 1e-10)
    
    # s1,s2,s3相对于s0归一化
    s0_for_norm = s0 + 1e-10
    s1_norm = s1 / s0_for_norm
    s2_norm = s2 / s0_for_norm
    s3_norm = s3 / s0_for_norm
    
    # 3. 保存结果
    normalized_data[..., 0] = s0_norm
    normalized_data[..., 1] = s1_norm
    normalized_data[..., 2] = s2_norm
    normalized_data[..., 3] = s3_norm
    
    return normalized_data




def normalize_stokes_simple(data):
    """
    归一化高光谱-偏振数据
    输入数据形状: [h, w, 21, 4]
    归一化规则:
        s0 (第4维的第0个元素) -> [0, 1]
        s1, s2, s3 (第4维的第1-3个元素) -> [-1, 1]
    """
    # 创建数据的副本以避免修改原始数据
    normalized_data = data.copy()

    means = np.mean(normalized_data, axis=(0, 1, 2))
    stds = np.std(normalized_data, axis=(0, 1, 2))
    
    out = np.zeros_like(normalized_data)
    for i in range(4):
        out[..., i] = (normalized_data[..., i] - means[i]) / stds[i]

    return out


def cal_4_state(data):
    h, w, c, s = data.shape
    logging.info(f'Input data shape: {data.shape}')
    states = np.zeros((h, w, c, 4), dtype=np.float32)
    s0, s1, s2, s3 = data[..., 0], data[..., 1], data[..., 2], data[..., 3]
    states[..., 0] = (s0 + s1)
    states[..., 1] = (s0 - s1)
    states[..., 2] = (s0 + s2)
    states[..., 3] = (s0 - s2)
    return states

def LoadTraining_npy_spec_polo(path, states=False, norm=False, add_reflect=False, Loading_dim=None):
    imgs = []
    scene_list = os.listdir(path)
    scene_list.sort()
    print('Training scenes:', len(scene_list))

    for i in range(len(scene_list)):
        scene_path = os.path.join(path, scene_list[i])


        if not scene_list[i].endswith('.npy'):
            continue


        img = np.load(scene_path)

        # img = img[:, :, :, [0, 7, 14, 20]].transpose(0, 1, 3, 2)
        img = img[:, :, :, :].transpose(0, 1, 3, 2)
        if norm:
            img = normalize_stokes(img)
        img = img.astype(np.float32)  # 转换为 float32

        if states:
            img = cal_4_state(img)

        img = img.astype(np.float32)
        if add_reflect:
            # img = add_90deg_polarized_reflection(img, min_size=50, max_size=200, intensity_range=(0, 1), num_patches_range=(0, 5))
            img = add_polarized_reflection_blurred(img, min_size=50, max_size=200, intensity_range=(0, 1), num_patches_range=(0, 5), random_angle=True, reflection_spectrum=reflect_led_spectrum)
        # img = (img - np.min(img)) / (np.max(img) - np.min(img))

        if Loading_dim == 'Polar':
            img = trans_specpolar_to_polar(img)
        if Loading_dim == 'Spec':
            img = img[ :, :, :, 0]  # S0

        imgs.append(img)
        print('Scene {} is loaded. {}'.format(i, scene_list[i]))

    return imgs




def random_load_spec_polo(path, max_images=None, shuffle=True, states=False, norm=False, add_reflect=False, Loading_dim=None):
    imgs = []
    scene_list = [f for f in os.listdir(path) if f.endswith('.npy')]  # 只保留.npy文件

    if shuffle:
        random.shuffle(scene_list)  # 随机打乱文件列表
    else:
        scene_list.sort()  # 不随机则按文件名排序

    print('Found {} .npy files. Loading {}...'.format(len(scene_list),
                                                      max_images if max_images is not None else 'all'))

    loaded_count = 0

    for i, scene_name in enumerate(scene_list):
        if max_images is not None and loaded_count >= max_images:
            break

        scene_path = os.path.join(path, scene_name)

        try:
            img = np.load(scene_path)
            img = img[:, :, :, :].transpose(0, 1, 3, 2)  # 调整维度  H W C S
            if norm:
                img = normalize_stokes(img)
            img = img.astype(np.float32)  # 转换为 float32
            
            if add_reflect:
                # img = add_90deg_polarized_reflection(img, min_size=50, max_size=200, intensity_range=(0, 1), num_patches_range=(0, 5))
                img = add_polarized_reflection_blurred(img, min_size=50, max_size=200, intensity_range=(0, 1), num_patches_range=(0, 5), random_angle=True, reflection_spectrum=reflect_led_spectrum)
            if states:
                img = cal_4_state(img)
            if Loading_dim == 'Polar':
                img = trans_specpolar_to_polar(img)
            if Loading_dim == 'Spec':
                img = img[ :, :, :, 0]  # S0
            imgs.append(img)
            loaded_count += 1
            print('Loaded {}/{}: {}'.format(loaded_count,
                                            max_images if max_images is not None else len(scene_list),
                                            scene_name))
        except Exception as e:
            print(f'Error loading {scene_name}: {str(e)}')
            continue

    print(
        f'Successfully loaded {len(imgs)}/{min(max_images, len(scene_list)) if max_images else len(scene_list)} files four states:{states}')
    return imgs



def trans_specpolar_to_polar(img):
    h, w, c, s = img.shape
    select_index = random.sample(range(c), 7)
    s0 = img[:, :, select_index[0], 0]
    s1 = img[:, :, select_index[1], 0] - img[:, :, select_index[2], 0]
    s2 = img[:, :, select_index[3], 0] - img[:, :, select_index[4], 0]
    s3 = img[:, :, select_index[5], 0] - img[:, :, select_index[6], 0]
    polar = np.zeros((h, w, 4))
    polar[:, :, 0] = s0
    polar[:, :, 1] = s1
    polar[:, :, 2] = s2
    polar[:, :, 3] = s3
    return polar[..., np.newaxis].transpose(0, 1, 3, 2)


def trans_four_to_six(img, norm=False):
    h, w, c, s = img.shape
    six = np.zeros((h, w, c, 6))
    
    # 使用更清晰的索引方式
    S0 = img[..., 0]  # 总强度
    S1 = img[..., 1]  # 水平-垂直分量
    S2 = img[..., 2]  # 45°-135°分量  
    S3 = img[..., 3]  # 圆偏振分量
    
    six[..., 0] = (S0 + S1) 
    six[..., 1] = (S0 - S1) 
    six[..., 2] = (S0 + S2) 
    six[..., 3] = (S0 - S2) 
    six[..., 4] = (S0 + S3) 
    six[..., 5] = (S0 - S3)



    
    # 如果需要进行分通道归一化
    if norm:
        six = channel_wise_normalize(six)
    
    return six


def trans_six_to_four(img, norm=False):
    h, w, c, s = img.shape
    four = np.zeros((h, w, c, 4))
    

    H = img[..., 0]
    V = img[..., 1]
    A = img[..., 2]
    D = img[..., 3]
    L = img[..., 4]
    R = img[..., 5]
    
    four[..., 0] = (H + V) / 2
    four[..., 1] = (H - V) / 2
    four[..., 2] = (A - D) / 2
    four[..., 3] = (L - R) / 2


    
    return four

def channel_wise_normalize(data, epsilon=1e-8):
    """
    每个偏振通道独立归一化到[0,1]
    """
    normalized = np.zeros_like(data)
    for i in range(data.shape[-1]):
        channel = data[..., i]
        min_val = np.min(channel)
        max_val = np.max(channel)
        
        if max_val > min_val + epsilon:
            normalized[..., i] = (channel - min_val) / (max_val - min_val)
    
    return normalized




def random_load_spec_polo_six(path, max_images=None, shuffle=True, states=False, norm=False):
    imgs = []
    scene_list = [f for f in os.listdir(path) if f.endswith('.npy')]  # 只保留.npy文件

    if shuffle:
        random.shuffle(scene_list)  # 随机打乱文件列表
    else:
        scene_list.sort()  # 不随机则按文件名排序

    print('Found {} .npy files. Loading {}...'.format(len(scene_list),
                                                      max_images if max_images is not None else 'all'))

    loaded_count = 0

    for i, scene_name in enumerate(scene_list):
        if max_images is not None and loaded_count >= max_images:
            break

        scene_path = os.path.join(path, scene_name)

        try:
            img = np.load(scene_path)
            img = img[:, :, :, :].transpose(0, 1, 3, 2)  # 调整维度

            img = trans_four_to_six(img, norm)

            
            imgs.append(img)
            loaded_count += 1
            print('Loaded {}/{}: {}'.format(loaded_count,
                                            max_images if max_images is not None else len(scene_list),
                                            scene_name))
        except Exception as e:
            print(f'Error loading {scene_name}: {str(e)}')
            continue

    print(
        f'Successfully loaded {len(imgs)}/{min(max_images, len(scene_list)) if max_images else len(scene_list)} files four states:{states}')
    return imgs






def shuffle_crop_npy_spec_polo(train_data, batch_size, crop_size, argument=True, seed=None):   # list x h x w x c
    if seed is not None:
        np.random.seed(seed)
    if argument:
        h, w, c, s = train_data[0].shape

        gt_batch = []
        # The first half data use the original data.
        index = np.random.choice(range(len(train_data)), batch_size//2)
        processed_data = np.zeros((batch_size//2, crop_size, crop_size, c, s), dtype=np.float32)
        for i in range(batch_size//2):
            img = train_data[index[i]]
            h, w, _, _ = img.shape
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            processed_data[i, :, :, :, :] = img[x_index:x_index + crop_size, y_index:y_index + crop_size, :, :]
        processed_data = torch.from_numpy(np.transpose(processed_data, (0, 3, 4, 1, 2))).cuda().float()
        for i in range(processed_data.shape[0]):
            gt_batch.append(arguement_1_spec_polo(processed_data[i]))

        # The other half data use splicing.
        processed_data = np.zeros((4, crop_size//2, crop_size//2, c, s), dtype=np.float32)
        for i in range(batch_size - batch_size // 2):
            sample_list = np.random.randint(0, len(train_data), 4)
            for j in range(4):
                x_index = np.random.randint(0, h-crop_size//2)
                y_index = np.random.randint(0, w-crop_size//2)
                processed_data[j] = train_data[sample_list[j]][x_index:x_index+crop_size//2,y_index:y_index+crop_size//2, :, :]
            gt_batch_2 = torch.from_numpy(np.transpose(processed_data, (0, 3, 4, 1, 2))).cuda()  # [4,28,128,128]
            gt_batch.append(arguement_2_spec_polo(gt_batch_2, crop_size))
        gt_batch = torch.stack(gt_batch, dim=0)
        return gt_batch
    else:
        index = np.random.choice(range(len(train_data)), batch_size)
        processed_data = np.zeros((batch_size, crop_size, crop_size, 21), dtype=np.float32)
        for i in range(batch_size):
            h, w, _ = train_data[index[i]].shape
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            processed_data[i, :, :, :] = train_data[index[i]][x_index:x_index + crop_size, y_index:y_index + crop_size, :]
        gt_batch = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2)))
        return gt_batch


def arguement_1_spec_polo(x):
    """
    :param x: c,h,w
    :return: c,h,w
    """
    rotTimes = random.randint(0, 3)
    vFlip = random.randint(0, 1)
    hFlip = random.randint(0, 1)
    # Random rotation
    for j in range(rotTimes):
        x = torch.rot90(x, dims=(2, 3))
    # Random vertical Flip
    for j in range(vFlip):
        x = torch.flip(x, dims=(3,))
    # Random horizontal Flip
    for j in range(hFlip):
        x = torch.flip(x, dims=(2,))
    return x

def arguement_2_spec_polo(generate_gt, crop_size):
    c, s, h, w = generate_gt.shape[1], generate_gt.shape[2], crop_size, crop_size
    divid_point_h = h // 2
    divid_point_w = w // 2
    output_img = torch.zeros(c, s, h, w).cuda()
    output_img[:, :, :divid_point_h, :divid_point_w] = generate_gt[0]
    output_img[:, :, :divid_point_h, divid_point_w:] = generate_gt[1]
    output_img[:, :, divid_point_h:, :divid_point_w] = generate_gt[2]
    output_img[:, :, divid_point_h:, divid_point_w:] = generate_gt[3]
    return output_img







def gen_meas_torch(data_batch, mask3d_batch, Y2H=True, mul_mask=False):
    nC = data_batch.shape[1]
    temp = shift(mask3d_batch * data_batch, 2)  # 5 28 256 310
    meas = torch.sum(temp, 1)   # 5 256 310
    if Y2H:
        meas = meas / nC * 2
        H = shift_back(meas)   # 5 28 256 256
        if mul_mask:
            HM = torch.mul(H, mask3d_batch)
            return HM
        return H
    return meas

def shift(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col + (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, step * i:step * i + col] = inputs[:, i, :, :]
    return output

def shift_back(inputs, step=2):  # input [bs,256,310]  output [bs, 28, 256, 256]
    [bs, row, col] = inputs.shape
    nC = 28
    output = torch.zeros(bs, nC, row, col - (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, :] = inputs[:, :, step * i:step * i + col - (nC - 1) * step]
    return output

def gen_log(model_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    log_file = model_path + '/log.txt'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def init_mask(mask_path, mask_type, batch_size):
    mask3d_batch = generate_masks(mask_path, batch_size)
    if mask_type == 'Phi':
        shift_mask3d_batch = shift(mask3d_batch)
        input_mask = shift_mask3d_batch
    elif mask_type == 'Phi_PhiPhiT':
        Phi_batch, Phi_s_batch = generate_shift_masks(mask_path, batch_size)
        input_mask = (Phi_batch, Phi_s_batch)
    elif mask_type == 'Mask':
        input_mask = mask3d_batch
    elif mask_type == None:
        input_mask = None
    return mask3d_batch, input_mask

def init_meas(gt, mask, input_setting):
    if input_setting == 'H':
        input_meas = gen_meas_torch(gt, mask, Y2H=True, mul_mask=False)
    elif input_setting == 'HM':
        input_meas = gen_meas_torch(gt, mask, Y2H=True, mul_mask=True)
    elif input_setting == 'Y':
        input_meas = gen_meas_torch(gt, mask, Y2H=False, mul_mask=True)
    return input_meas

def checkpoint(model, epoch, model_path, logger):
    model_out_path = model_path + "/model_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    logger.info("Checkpoint saved to {}".format(model_out_path))


def generate_gaussian_image(num=10, crop_size=116, spectral_channels=21, polarization_channels=4):
    """
    为每张图片生成统一的光谱-偏振数据
    每张图片的四个偏振通道分别用一条高斯曲线表示
    空间分辨率：116x116
    光谱通道数：21
    偏振通道数：4 (s0, s1, s2, s3)
    
    返回：
        data: 形状为 (116, 116, 21, 4) 的numpy数组
    """
    data_list = []
    # 定义数据形状
    height, width = crop_size, crop_size
    # spectral_channels = 21
    # polarization_channels = 4
    min_width = 0.1
    
    # 创建空的数据数组
    for i in range(num):
        print(f'Loading:{i+1}/{num}')
        data = np.zeros((height, width, spectral_channels, polarization_channels))

        # 为每个偏振通道生成一条高斯光谱曲线（每张图片统一）
        # s0: 强度通道 (0-1)
        s0_amplitude = np.random.uniform(0.7, 1.0)
        s0_center = np.random.uniform(0, 20)
        s0_width = np.random.uniform(min_width, 3)

        # s1, s2, s3: 偏振通道 (-1到1)
        s1_amplitude = np.random.uniform(-1.0, 1.0)
        s1_center = np.random.uniform(0, 20)
        s1_width = np.random.uniform(min_width, 3)

        s2_amplitude = np.random.uniform(-1.0, 1.0)
        s2_center = np.random.uniform(0, 20)
        s2_width = np.random.uniform(min_width, 3)

        s3_amplitude = np.random.uniform(-1.0, 1.0)
        s3_center = np.random.uniform(0, 20)
        s3_width = np.random.uniform(min_width, 3)

        # 生成光谱轴 (0到20)
        x = np.linspace(0, 20, spectral_channels)

        # 计算高斯曲线
        s0_curve = s0_amplitude * np.exp(-((x - s0_center) ** 2) / (2 * s0_width ** 2))
        s1_curve = s1_amplitude * np.exp(-((x - s1_center) ** 2) / (2 * s1_width ** 2))
        s2_curve = s2_amplitude * np.exp(-((x - s2_center) ** 2) / (2 * s2_width ** 2))
        s3_curve = s3_amplitude * np.exp(-((x - s3_center) ** 2) / (2 * s3_width ** 2))

        # 确保值域正确
        s0_curve = np.clip(s0_curve, 0, 1)
        s1_curve = np.clip(s1_curve, -1, 1)
        s2_curve = np.clip(s2_curve, -1, 1)
        s3_curve = np.clip(s3_curve, -1, 1)

        # 将相同的光谱曲线复制到所有空间位置
        data[:, :, :, 0] = s0_curve  # s0通道所有像素相同
        data[:, :, :, 1] = s1_curve  # s1通道所有像素相同
        data[:, :, :, 2] = s2_curve  # s2通道所有像素相同
        data[:, :, :, 3] = s3_curve  # s3通道所有像素相同
        data_list.append(data)
    
    return data_list

def generate_gaussian_image_torch(batch_size=10, crop_size=118, spectral_channels=21, polarization_channels=4, device='cuda'):
    """
    PyTorch版本 - 生成高斯光谱偏振图像batch
    
    参数:
        batch_size: 批量大小
        crop_size: 空间分辨率
        spectral_channels: 光谱通道数
        polarization_channels: 偏振通道数 (s0, s1, s2, s3)
        device: 设备 ('cpu' 或 'cuda')
    
    返回:
        data: 形状为 (batch_size, spectral_channels, polarization_channels, crop_size, crop_size) 的tensor
    """
    # 预先生成光谱轴 (0到20)
    x = torch.linspace(0, 20, spectral_channels, device=device)
    
    # 一次性生成所有batch的参数
    amplitudes = torch.empty(batch_size, polarization_channels, device=device)
    amplitudes[:, 0] = torch.empty(batch_size, device=device).uniform_(0.8, 1.0)  # s0通道
    amplitudes[:, 1:] = torch.empty(batch_size, polarization_channels-1, device=device).uniform_(-1.0, 1.0)  # s1-s3通道
        # 为s1-s3通道生成振幅，要么在[0.7, 1.0]要么在[-1.0, -0.7]
    # for i in range(1, polarization_channels):
    #     # 随机选择正负区间
    #     sign_choice = torch.randint(0, 2, (batch_size,), device=device).float()
    #     # 生成基础振幅值 [0.7, 1.0]
    #     base_amplitude = torch.empty(batch_size, device=device).uniform_(0.7, 1.0)
    #     # 根据选择转换为正或负
    #     amplitudes[:, i] = torch.where(sign_choice == 0, base_amplitude, -base_amplitude)
    
    # # ... 其余代码保持不变 ...
    
    centers = torch.empty(batch_size, polarization_channels, device=device).uniform_(0, 20)
    widths = torch.empty(batch_size, polarization_channels, device=device).uniform_(0, 2)
    
    # 计算所有高斯曲线: (batch_size, polarization_channels, spectral_channels)
    curves = amplitudes.unsqueeze(-1) * torch.exp(
        -((x - centers.unsqueeze(-1)) ** 2) / (2 * widths.unsqueeze(-1) ** 2)
    )
    
    # 确保值域正确
    curves[:, 0, :] = torch.clamp(curves[:, 0, :], 0, 1)  # s0通道
    curves[:, 1:, :] = torch.clamp(curves[:, 1:, :], -1, 1)  # s1-s3通道
    
    # 调整维度顺序: (batch_size, spectral_channels, polarization_channels)
    curves = curves.permute(0, 2, 1)
    
    # 扩展到空间维度: (batch_size, spectral_channels, polarization_channels, crop_size, crop_size)
    data = curves.unsqueeze(-1).unsqueeze(-1)  # 添加空间维度
    data = data.expand(batch_size, spectral_channels, polarization_channels, crop_size, crop_size)
    
    return data




def cal_deg(data):
    m0 = data[:, 0, :, :]
    m1 = data[:, 1, :, :]
    m2 = data[:, 2, :, :]
    m3 = data[:, 3, :, :]


    theta_rad = 0.5 * np.arctan2(m2, m1)
    theta_deg = np.rad2deg(theta_rad) % 180  # 转换为度并限制在0-180之间

    docp = np.divide(m3, m0, out=np.zeros_like(m0), where=(m0 != 0))

    return theta_deg, docp

def stretch_band(band, lower_percent=2, upper_percent=98):
    """
    对单波段图像进行对比度拉伸
    """
    # 计算指定百分位的值
    low_val = np.percentile(band, lower_percent)
    high_val = np.percentile(band, upper_percent)
    
    # 线性拉伸
    band_stretched = (band - low_val) / (high_val - low_val)
    
    # 裁剪到[0, 1]范围
    band_stretched = np.clip(band_stretched, 0, 1)
    
    return band_stretched

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
    
    if hyperspectral_data.shape[2] == 18:
        cie_x = cie_x[3:]
        cie_y = cie_y[3:]
        cie_z = cie_z[3:]


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


def trans_multi_2_rgb_v2(hyperspectral_data):
    """
    将对齐到 450-650nm (间隔10nm, 共21个通道) 的高光谱数据转换为 sRGB 图像
    输入: hyperspectral_data [H, W, C]  C=21
    输出: rgb_image [H, W, 3]   值范围 [0,1]
    """

    # 对应波长: 450, 460, ..., 650 (共21个)
    cie_wavelength = np.arange(450, 651, 10)

    # CIE 1931 2° 标准观察者 色彩匹配函数 (CMF)，与上面波长对齐
    cie_x = np.array([0.3362, 0.2908, 0.1954, 0.0956, 0.0320,
                      0.0049, 0.0093, 0.0633, 0.1655, 0.2904,
                      0.4334, 0.5945, 0.7621, 0.9163, 1.0263,
                      1.0622, 1.0026, 0.8544, 0.6424, 0.4479,
                      0.2835])

    cie_y = np.array([0.0380, 0.0600, 0.0910, 0.1390, 0.2080,
                      0.3230, 0.5030, 0.7100, 0.8620, 0.9540,
                      0.9950, 0.9950, 0.9520, 0.8700, 0.7570,
                      0.6310, 0.5030, 0.3810, 0.2650, 0.1750,
                      0.1070])

    cie_z = np.array([1.7721, 1.6692, 1.2876, 0.8130, 0.4652,
                      0.2720, 0.1582, 0.0782, 0.0422, 0.0203,
                      0.0087, 0.0039, 0.0021, 0.0017, 0.0011,
                      0.0008, 0.0003, 0.0002, 0.0000, 0.0000,
                      0.0000])

    if hyperspectral_data.shape[2] == 18:
        cie_x = cie_x[3:]
        cie_y = cie_y[3:]
        cie_z = cie_z[3:]

    # Step1: 高光谱投影到 XYZ
    X = np.tensordot(hyperspectral_data, cie_x, axes=([2], [0]))
    Y = np.tensordot(hyperspectral_data, cie_y, axes=([2], [0]))
    Z = np.tensordot(hyperspectral_data, cie_z, axes=([2], [0]))

    xyz_image = np.stack([X, Y, Z], axis=-1)  # [H,W,3]

    # Step2: XYZ → sRGB (D65, linear RGB)
    M = np.array([[ 3.2406, -1.5372, -0.4986],
                  [-0.9689,  1.8758,  0.0415],
                  [ 0.0557, -0.2040,  1.0570]])

    rgb_linear = np.tensordot(xyz_image, M.T, axes=([2],[0]))

    # Step3: Clip & Gamma 校正 (sRGB gamma≈2.2)
    rgb_linear = np.clip(rgb_linear, 0, None)  # 去掉负值
    rgb_image = np.power(rgb_linear, 1/2.2)

    # Step4: 归一化到 [0,1]
    rgb_image = rgb_image / np.max(rgb_image)

    return rgb_image



def gen_full_img(test_gt, model, filter_matrix_test):
    test_data = np.stack(test_gt, axis=0).transpose(0, 3, 4, 1, 2)  # 5 x 512 612 4 4
    bs = test_data.shape[0]
    data_list = []
    for i in range(1, 6):
        for j in range(1, 6):
            test_gt = test_data[:, :, :, 100*(i-1):100*i, 100*(j-1):100*j]

            real_gt = test_gt

            test_input = torch.tensor(test_gt).cuda()

            # test_input /= opt.ratio
            # test_input[:, :, 0, :, :] / opt.ratio

            # input_meas = init_meas(test_gt, mask3d_batch_test, opt.input_setting)
            model.eval()

            with torch.no_grad():
                model_out = model(test_input, filter_matrix_test)
                model_out = model_out.view(bs, 21, 4, 100, 100)
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


    model_out = full_image.detach().cpu().numpy()

    return model_out


def add_90deg_polarized_reflection(data, min_size=50, max_size=200, intensity_range=(0, 1), num_patches_range=(0, 5), random_angle=False):
    """
    在数据集中添加随机的线偏振反光斑块
    
    参数:
    - data: 输入数据, 形状 (512, 612, 4, 21)
    - min_size: 反光斑块最小尺寸
    - max_size: 反光斑块最大尺寸
    - intensity_range: 反光强度范围 (相对于原始信号)
    - num_patches_range: 反光斑块数量范围
    - random_angle: 如果为True，则生成0-180°随机的线偏振角度；如果为False，则使用90°线偏振
    
    返回:
    - 添加反光后的数据
    - 反光掩码 (用于可视化)
    """
    data = data.transpose(0, 1, 3, 2)  # 调整维度  H W C S
    h, w, _, spectral_dim = data.shape
    result = data.copy()
    reflection_mask = np.zeros((h, w), dtype=bool)
    
    # 随机生成反光斑块数量
    num_patches = random.randint(num_patches_range[0], num_patches_range[1])
    
    for _ in range(num_patches):
        # 随机生成斑块尺寸
        patch_h = random.randint(min_size, max_size)
        patch_w = random.randint(min_size, max_size)
        
        # 随机生成位置 (确保在图像范围内)
        x = random.randint(0, w - patch_w - 1)
        y = random.randint(0, h - patch_h - 1)
        
        # 随机决定是否旋转 (创建倾斜矩形)
        rotation_angle = random.uniform(0, 45) if random.random() > 0.5 else 0
        
        # 创建斑块掩码
        patch_mask = create_patch_mask(h, w, x, y, patch_w, patch_h, rotation_angle)
        reflection_mask |= patch_mask
        
        # 随机生成反光强度
        intensity = random.uniform(intensity_range[0], intensity_range[1])
        
        # 随机生成偏振角度（如果启用）
        if random_angle:
            polarization_angle = random.uniform(0, 180)  # 0-180度随机角度
        else:
            polarization_angle = 90  # 默认90度
        
        # 将角度转换为弧度
        angle_rad = np.radians(polarization_angle)
        
        # 计算对应角度的Stokes参数
        # 对于角度θ的线偏振: [S0, S1, S2, S3] = [1, cos(2θ), sin(2θ), 0]
        reflection_stokes = np.array([
            1.0,                           # S0: 总强度
            np.cos(2 * angle_rad),         # S1: 水平-垂直偏振分量
            np.sin(2 * angle_rad),         # S2: 45°-135°偏振分量
            0.0                            # S3: 圆偏振分量为0
        ])
        
        # 为每个光谱通道添加线偏振反光
        for spec_idx in range(spectral_dim):
            # 将反光叠加到原始数据
            for stokes_idx in range(4):
                result[:, :, stokes_idx, spec_idx][patch_mask] += (
                    reflection_stokes[stokes_idx] * intensity * 
                    np.mean(data[:, :, 0, spec_idx][patch_mask])  # 基于原始强度缩放
                )
    
    return result.transpose(0, 1, 3, 2)  # 调整维度  H W C S

def create_patch_mask(h, w, x, y, patch_w, patch_h, rotation_angle=0):
    """
    创建矩形或倾斜矩形斑块掩码
    """
    mask = np.zeros((h, w), dtype=bool)
    
    if rotation_angle == 0:
        # 不旋转的矩形
        mask[y:y+patch_h, x:x+patch_w] = True
    else:
        # 创建倾斜矩形 (使用简单的旋转)
        center_x = x + patch_w // 2
        center_y = y + patch_h // 2
        
        # 创建旋转后的边界点
        angle_rad = np.radians(rotation_angle)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        # 检查每个像素是否在旋转后的矩形内
        for i in range(y, y + patch_h):
            for j in range(x, x + patch_w):
                # 相对于中心点的坐标
                rel_x = j - center_x
                rel_y = i - center_y
                
                # 旋转回原坐标系
                rot_x = rel_x * cos_angle + rel_y * sin_angle
                rot_y = -rel_x * sin_angle + rel_y * cos_angle
                
                # 检查是否在原始矩形内
                if (abs(rot_x) <= patch_w / 2 and abs(rot_y) <= patch_h / 2):
                    mask[i, j] = True
    
    return mask


def add_polarized_reflection_blurred(
    data, min_size=50, max_size=200, 
    intensity_range=(0, 1), num_patches_range=(0, 5),
    random_angle=False, 
    blur_sigma_range=(5, 20),
    # !!! 新增参数: 确定的反射光谱曲线 (形状应为 (spectral_dim,))
    reflection_spectrum=None 
):
    """
    在数据集中添加随机的线偏振反光斑块，具有模糊效果和确定的光谱曲线。
    
    参数:
    - data: 输入数据, 形状 (H, W, 4, S)
    - reflection_spectrum: 确定的反射光谱曲线，形状 (S,)。用于缩放每个波段的反射强度。
                           如果为 None，则默认为白光（所有波段权重为 1）。
    - 其他参数同上...
    
    返回:
    - 添加反光后的数据 (H, W, 4, S)
    - 反光权重图 (H, W) (用于可视化模糊效果)
    """
    data = data.transpose(0, 1, 3, 2)
    h, w, _, spectral_dim = data.shape
    result = data.copy()
    total_reflection_weight = np.zeros((h, w), dtype=np.float32) 
    
    # --- A. 预处理和验证反射光谱 ---
    if reflection_spectrum is None:
        # 如果没有提供，默认使用白光 (所有通道权重为 1)
        reflection_spectrum = np.ones(spectral_dim, dtype=data.dtype)
    else:
        # 验证输入光谱维度
        if reflection_spectrum.shape != (spectral_dim,):
            raise ValueError(f"reflection_spectrum 维度必须是 ({spectral_dim},), 但收到了 {reflection_spectrum.shape}")
        # 归一化光谱曲线，确保其最大值为 1.0，以便 intensity_range 控制峰值强度
        max_spec = np.max(reflection_spectrum)
        if max_spec > 0:
             reflection_spectrum = reflection_spectrum / max_spec
    
    # 随机生成反光斑块数量
    num_patches = random.randint(num_patches_range[0], num_patches_range[1])
    
    for _ in range(num_patches):
        # 1. 随机生成斑块尺寸、位置和旋转角度 (逻辑不变)
        patch_h = random.randint(min_size, max_size)
        patch_w = random.randint(min_size, max_size)
        x = random.randint(0, w - patch_w - 1)
        y = random.randint(0, h - patch_h - 1)
        rotation_angle = random.uniform(0, 45) if random.random() > 0.5 else 0
        
        # 使用改进后的函数创建浮点型硬掩码 (1.0/0.0)
        # 假设 create_patch_mask 已经返回 np.float32
        patch_mask_hard = create_patch_mask_blur(h, w, x, y, patch_w, patch_h, rotation_angle)
        
        # 2. --- 引入模糊效果的关键步骤 (逻辑不变) ---
        sigma = random.uniform(blur_sigma_range[0], blur_sigma_range[1])
        patch_weight = cv2.GaussianBlur(patch_mask_hard, (0, 0), sigma)
        
        max_val = patch_weight.max()
        if max_val > 0:
            patch_weight /= max_val
        else:
            continue
        
        # 3. 随机生成反光强度（控制总峰值强度）和偏振角度
        peak_intensity_factor = random.uniform(intensity_range[0], intensity_range[1])
        
        if random_angle:
            polarization_angle = random.uniform(0, 180)
        else:
            polarization_angle = 90
            
        angle_rad = np.radians(polarization_angle)
        
        reflection_stokes = np.array([
            1.0, np.cos(2 * angle_rad), np.sin(2 * angle_rad), 0.0
        ])
        
        # 4. 将反光叠加到原始数据（使用权重图 W）
        
        # 确定需要计算原始平均 S0 强度的区域
        patch_mask_hard_bool = patch_mask_hard.astype(bool)
        if not np.any(patch_mask_hard_bool):
            continue
            
        # 计算该斑块的 S0 平均值（在所有波段上取平均，得到一个标量）
        # 这个值作为整体亮度的基准，使反光强度与图像的整体亮度相关
        original_s0_mean_scalar = np.mean(data[:, :, 0, :][patch_mask_hard_bool]) 
        
        # 叠加循环
        for spec_idx in range(spectral_dim):
            
            # !!! 关键修改: 使用确定的反射光谱曲线来计算该波段的最终强度 !!!
            
            # 1. 强度基准 (与原始 S0 相关的绝对亮度)
            # 2. 强度因子 (intensity_range 决定的随机缩放)
            # 3. 光谱权重 (reflection_spectrum 决定的波段比例)
            
            reflection_amplitude = (
                original_s0_mean_scalar * peak_intensity_factor * reflection_spectrum[spec_idx]
            )

            # 权重矩阵 W: (h, w)
            W = patch_weight 
            
            for stokes_idx in range(4):
                C = reflection_stokes[stokes_idx] # Stokes分量权重
                
                # 叠加量 = Stokes分量 * 反射光在该通道的强度 * 空间权重
                result[:, :, stokes_idx, spec_idx] += C * reflection_amplitude * W
        
        # 累积模糊后的权重图
        total_reflection_weight += patch_weight * peak_intensity_factor

    # 返回结果数据和模糊权重图
    return result.transpose(0, 1, 3, 2)


def create_patch_mask_blur(h, w, x, y, patch_w, patch_h, rotation_angle=0):
    """
    创建矩形或倾斜矩形斑块的浮点型硬掩码 (1.0 在内, 0.0 在外)
    """
    # 确保返回 float32 类型，以便后续模糊处理
    mask = np.zeros((h, w), dtype=np.float32) 
    
    if rotation_angle == 0:
        # 不旋转的矩形
        mask[y:y+patch_h, x:x+patch_w] = 1.0
    else:
        # 创建倾斜矩形 (使用你原来的旋转逻辑)
        center_x = x + patch_w / 2
        center_y = y + patch_h / 2
        
        angle_rad = np.radians(rotation_angle)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        # 检查每个像素是否在旋转后的矩形内
        for i in range(h):
            for j in range(w):
                # 仅在可能包含斑块的区域内计算，以提高效率 (可选)
                
                rel_x = j - center_x
                rel_y = i - center_y
                
                # 旋转回原坐标系
                rot_x = rel_x * cos_angle + rel_y * sin_angle
                rot_y = -rel_x * sin_angle + rel_y * cos_angle
                
                # 检查是否在原始矩形内
                if (abs(rot_x) <= patch_w / 2 and abs(rot_y) <= patch_h / 2):
                    mask[i, j] = 1.0
    
    return mask