from unittest import result
from architecture.MST_polo import add_gaussian_noise
from torch import nn as nn
from tqdm import tqdm
from matplotlib import pyplot as plt
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
# from architecture.MST_duizhao import MST_duizhao

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

seed = 529
set_seed(seed)


def torch_psnr(img, ref):
    # 保证在0-1范围
    img = img.clamp(0, 1) * 255.0
    ref = ref.clamp(0, 1) * 255.0

    mse = torch.mean((img - ref) ** 2)  # 对整个 batch 求MSE
    psnr = 10 * torch.log10((255.0 * 255.0) / (mse + 1e-10))
    return psnr



torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')



if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")

print("Current working directory:", os.getcwd())
print("Mask file path:", os.path.join(opt.mask_path, 'mask.mat'))

# init mask
mask3d_batch_train, input_mask_train = init_mask(opt.mask_path, opt.input_mask, opt.batch_size)
mask3d_batch_test, input_mask_test = init_mask(opt.mask_path, opt.input_mask, 10)

# dataset
# train_set = LoadTraining_npy_spec_polo(opt.data_path)
train_set = random_load_spec_polo(opt.data_path, opt.select_num, norm=True)
test_data = LoadTraining_npy_spec_polo(opt.test_path, norm=True)

# saving path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
result_path = 'ablation_test'+ '/' + opt.save_name + '/result/'  # train_t_mask 为1_1
model_path = 'ablation_test' + '/' + opt.save_name + '/model/'


if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)

# model
# if opt.method=='hdnet':
#     model, FDL_loss = model_generator(opt.method, opt.pretrained_model_path).cuda()
# else:
#     model = model_generator(opt.method, opt.pretrained_model_path).cuda()

model = MST_duizhao(dim=21*4, stage=2, num_blocks=[4, 7, 5], need_noise=True, num_sp=5, need_mask_atten=False).cuda()
model = MST_with_resnet(dim=21*4, stage=2, num_blocks=[2, 2, 2], need_noise=False, num_sp=5, need_mask_atten=False).cuda()

# model = MST(dim=4, stage=2, num_blocks=[4, 7, 5], need_noise=opt.need_noise, num_sp=5, need_mask_atten=False).cuda()
filter_path = 'mask/93_hand_bd_10_10.npy' #更换为自己矩阵
filter_path = 'mask/99_100_100_full_matrix.npy' #更换为自己矩阵


print(f'Now using {filter_path} ')


filter_matrix = torch.from_numpy(np.load(filter_path)).unsqueeze(0).repeat(opt.batch_size, 1, 1, 3, 3).cuda().float()
filter_matrix_test = torch.from_numpy(np.load(filter_path)).unsqueeze(0).repeat(1, 1, 1, 7, 7).cuda().float()


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=opt.learning_rate,
    betas=(0.9, 0.999),
    # weight_decay=1e-5  # 添加权重衰减，从1e-5开始尝试
)


if opt.scheduler=='MultiStepLR':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
elif opt.scheduler=='CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epoch, eta_min=1e-6)
mse = torch.nn.MSELoss().cuda()
L1_loss = torch.nn.L1Loss().cuda()
L2_loss = torch.nn.MSELoss().cuda()



def train(epoch, logger):
    psnr_list = []
    ssim_list = []
    epoch_loss = 0
    begin = time.time()
    batch_num = int(np.floor(opt.epoch_sam_num / opt.batch_size))
    


    # 使用tqdm显示进度条
    for i in tqdm(range(batch_num), desc=f'Epoch {epoch}', unit='batch'):
        mask_x = random.randint(0, 299-opt.crop_size)
        mask_y = random.randint(0, 299-opt.crop_size)
        filter_matrix_shuffle = filter_matrix[:, :, :, mask_x:mask_x+opt.crop_size, mask_y:mask_y+opt.crop_size]
        
        noisy_filter = add_gaussian_noise(filter_matrix_shuffle, 0.15)
        gt_batch = shuffle_crop_npy_spec_polo(train_set, opt.batch_size, opt.crop_size)  # 9 4 4 200 200
        spec_data = gt_batch.clone()
        optimizer.zero_grad()
        real_gt = spec_data
        input = spec_data


        # cur_filter_matrix = add_gaussian_noise(filter_matrix).cuda().float()
        model_out = model(input, noisy_filter)  # 9 16 200 200
        model_out = model_out.view(opt.batch_size, 21, 4, opt.crop_size, opt.crop_size)
        # loss = torch.sqrt(mse(model_out, gt))



        pre_loss_norm = L1_loss(model_out, real_gt)
        pre_loss = L2_loss(model_out, real_gt)


 
        loss = pre_loss_norm 



        epoch_loss += loss.data
        loss.backward()
        optimizer.step()

    end = time.time()

    for k in range(model_out.shape[0]):

        psnr_train= torch_psnr(model_out[k, :, 0, :, :], input[k, :, 0, :, :])
        ssim_train = torch_ssim(model_out[k, :, 0, :, :], input[k, :, 0, :, :])
        psnr_list.append(psnr_train.detach().cpu().numpy())
        ssim_list.append(ssim_train.detach().cpu().numpy())

    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    logger.info("===> Epoch {} Complete: Avg. Loss: {:.6f} time: {:.2f}".
                format(epoch, epoch_loss / batch_num, (end - begin)))
    logger.info('===> train psnr = {:.2f}, ssim = {:.3f}'
                .format(psnr_mean, ssim_mean))

    return 0

def test(epoch, logger):
    global filter_matrix_test
    noisy_filter_test = add_gaussian_noise(filter_matrix_test, 0.15)
    noisy_filter_test = filter_matrix_test

    psnr_list, ssim_list = [], []
    dop_err_list, cpf_err_list = [], []
    pred_list, truth_list = [], []

    # 准备测试数据
    test_gt_all = np.stack(test_data, axis=0).transpose(0, 3, 4, 1, 2)  
    test_gt_all = test_gt_all[5:]  # 10 21 4 512 612

    if opt.method == 'SSP' or opt.method == 'SPECAT':
        test_gt_all = test_gt_all[:, :, :, :512, :608] # window size 8

    N = test_gt_all.shape[0]
    H, W = test_gt_all.shape[-2], test_gt_all.shape[-1]
    noisy_filter_test = noisy_filter_test[:, :, :, :H, :W]

    model.eval()
    with torch.no_grad():
        for k in range(N):
            gt_np = test_gt_all[k]  # 形状: 21, 4, H, W
            gt_tensor = torch.tensor(gt_np[np.newaxis, ...]).cuda()  
            out_tensor = model(gt_tensor, noisy_filter_test)  
            out_tensor = out_tensor.view(1, 21, 4, gt_np.shape[-2], gt_np.shape[-1])
            out_np = out_tensor[0].cpu().numpy()  # 形状: 21, 4, H, W

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


    logger.info('Testing Epoch {}: PSNR={:.2f}, SSIM={:.3f}, DoP MSE={:.4f}, CPF MSE={:.4f}'
                .format(epoch, psnr_mean, ssim_mean, dop_err_mean, cpf_err_mean))

    model.train()
    return pred_list, truth_list, psnr_list, ssim_list, psnr_mean, ssim_mean, dop_err_mean, cpf_err_mean


def main():
    set_seed(seed)
    logger = gen_log(model_path)
    logger.info("Learning rate:{}, batch_size:{}.\n".format(opt.learning_rate, opt.batch_size))
    psnr_max = 0
    for epoch in range(1, opt.max_epoch + 1):
        train(epoch, logger)
        (pred, truth, psnr_all, ssim_all, psnr_mean, ssim_mean, dop_err_mean, cpf_err_mean) = test(epoch, logger)
        scheduler.step()
        if (psnr_mean > psnr_max and psnr_mean != np.inf):
            psnr_max = psnr_mean 
            name = result_path + '/' + 'Test_%d_%.2f_%.3f_%.4f_%.4f' % (epoch, psnr_max, ssim_mean, dop_err_mean, cpf_err_mean)
            # scio.savemat(name, {'truth': truth, 'pred': pred, 'psnr_list': psnr_all, 'ssim_list': ssim_all})
            # draw_test_images_only_spec(truth, pred, name, psnr_all, ssim_all)
            draw_test_images(truth, pred, name, psnr_all, ssim_all)
            checkpoint(model, epoch, model_path, logger)

if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main()




