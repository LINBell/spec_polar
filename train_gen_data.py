from torch import nn as nn
from tqdm import tqdm
from matplotlib import pyplot as plt
from architecture import *
# from simulation.train_code.temp_test import stack_21
from architecture.MST_polo import add_gaussian_noise
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

# def torch_psnr(img, ref):  #  c s h w 4 4 200 200
#     # 转换到0-255范围并取整
#     img = (img * 256).round().clamp(0, 255)
#     ref = (ref * 256).round().clamp(0, 255)

#     nC = img.shape[0]
#     psnr = 0.0
#     epsilon = 1e-10  # 防止除零

#     for i in range(nC):
#         mse = torch.mean((img[i] - ref[i]) ** 2)
#         psnr += 10 * torch.log10((255 ** 2) / (mse + epsilon))

#     return psnr / nC
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
# print(f"opt.gpu_id: {opt.gpu_id}, type: {type(opt.gpu_id)}")

# print(f"Current device: {torch.cuda.current_device()}")  # 当前使用的 GPU 编号
# print(f"Device name: {torch.cuda.get_device_name(int(opt.gpu_id))}")  # GPU 2 的名称


if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")

print("Current working directory:", os.getcwd())
print("Mask file path:", os.path.join(opt.mask_path, 'mask.mat'))

# init mask
mask3d_batch_train, input_mask_train = init_mask(opt.mask_path, opt.input_mask, opt.batch_size)
mask3d_batch_test, input_mask_test = init_mask(opt.mask_path, opt.input_mask, 10)

# dataset
# train_set = LoadTraining_npy_spec_polo(opt.data_path)
# train_set = random_load_spec_polo(opt.data_path, opt.select_num)
# test_data = LoadTraining_npy_spec_polo(opt.test_path)

# saving path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
result_path = 'GEN'+ '/' + opt.save_name + '/result/'  # train_t_mask 为1_1
model_path = 'GEN' + '/' + opt.save_name + '/model/'


if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)

# model
if opt.method=='hdnet':
    model, FDL_loss = model_generator(opt.method, opt.pretrained_model_path).cuda()
else:
    model = model_generator(opt.method, opt.pretrained_model_path).cuda()

# pre_model_path = '/data8T/lzj/MST_Spectral/simulation/train_code/exp/mst_l/train_T_mask_10_random/model/model_epoch_255.pth'
# model.load_state_dict(torch.load(pre_model_path))
# print("Successful loaded")


# filter_path = 'transpose_matrix.pth'
# filter_matrix = torch.load(filter_path).unsqueeze(0).repeat(opt.batch_size, 1, 1, 1, 1).cuda()
# filter_matrix_test = torch.load(filter_path).unsqueeze(0).repeat(5, 1, 1, 1, 1).cuda()

# filter_path = '21_trans_matrix.npy'
# filter_path = 'second_select_25_matrix.npy' #更换为自己矩阵
# filter_path = 'mask/1_1_matrix.npy' #更换为自己矩阵
# filter_path = 'mask/93_hand_bd_stokes.npy' #更换为自己矩阵
filter_path = 'mask/190_190_gen_data_train_curve.npy' #更换为自己矩阵

print(f'Now using {filter_path} ')
filter_matrix = torch.from_numpy(np.load(filter_path)).unsqueeze(0).repeat(opt.batch_size, 1, 1, 1, 1).cuda().float()
filter_matrix_test = torch.from_numpy(np.load(filter_path)).unsqueeze(0).repeat(10, 1, 1, 1, 1).cuda().float()
filter_matrix = filter_matrix[:, :, :, :opt.crop_size, :opt.crop_size]
filter_matrix_test = filter_matrix_test[:, :, :, :opt.crop_size, :opt.crop_size]



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
out_spec_channel = 21*4




# test_set = generate_gaussian_image(10, opt.crop_size*4, 21, 4)
# test_set = generate_gaussian_image_torch_fixed(10, opt.crop_size, 21)
test_set = generate_gaussian_image_torch(10, opt.crop_size*2, 21, 4)


def train(epoch, logger):
    global filter_matrix
    psnr_list = []
    ssim_list = []
    epoch_loss = 0
    begin = time.time()
    batch_num = int(np.floor(opt.epoch_sam_num / opt.batch_size))
    noisy_filter = add_gaussian_noise(filter_matrix, 0.16)


    # 使用tqdm显示进度条
    for i in tqdm(range(batch_num), desc=f'Epoch {epoch}', unit='batch'):
        # train_set = generate_gaussian_image(opt.batch_size, opt.crop_size*4, 21, 4)

        # gt_batch = shuffle_crop_npy_spec_polo(train_set, opt.batch_size, opt.crop_size, argument=False)  # 9 4 4 200 200
        
        train_set = generate_gaussian_image_torch(opt.batch_size, opt.crop_size,21, 4)
        # train_set = generate_gaussian_image_torch_fixed(opt.batch_size, opt.crop_size, 21)
        gt_batch = train_set

        spec_data = gt_batch.clone()
        optimizer.zero_grad()
        real_gt = spec_data
        input = spec_data




        model_out = model(input, noisy_filter)  # 9 16 200 200
        model_out = model_out.view(opt.batch_size, 21, 4, opt.crop_size, opt.crop_size)
        # loss = torch.sqrt(mse(model_out, gt))



        pre_loss_norm = L1_loss(model_out, real_gt)
        pre_loss = L2_loss(model_out, real_gt)


        grad_spec_pre = model_out[:, 1:21, :, :, :] - model_out[:, 0:21 - 1, :, :, :]
        grad_spec_label = real_gt[:, 1:21, :, :, :] - real_gt[:, 0:21 - 1, :, :, :]
        spec_loss = L2_loss(grad_spec_pre, grad_spec_label)


        # loss = pre_loss + 1e-1 * spec_loss + pre_loss_norm
        # loss = pre_loss + pre_loss_norm
        loss = pre_loss
        # loss = pre_loss 



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
    noisy_filter_test = add_gaussian_noise(filter_matrix_test, 0.16)
    psnr_list, ssim_list = [], []
    test_gt = np.stack(test_set.detach().cpu().numpy(), axis=0).astype(np.float32)   # 5 x 512 612 4 4
    
    # 随机裁剪每张图片的opt.crop_size区域
    N, C, S, H, W = test_gt.shape
    crop_h, crop_w = opt.crop_size, opt.crop_size
    cropped_imgs = []
    for i in range(N):
        h_start = np.random.randint(0, H - crop_h + 1)
        w_start = np.random.randint(0, W - crop_w + 1)
        cropped = test_gt[i, :, :, h_start:h_start+crop_h, w_start:w_start+crop_w]
        cropped_imgs.append(cropped)
    test_gt = np.stack(cropped_imgs, axis=0)
    test_gt = torch.tensor(test_gt).cuda().float()
    test_input = test_gt

    model.eval()
    begin = time.time()
    with torch.no_grad():
        model_out = model(test_input, noisy_filter_test)
        model_out = model_out.view(10, 21, 4, opt.crop_size, opt.crop_size)
        model_out = model_out
    end = time.time()
    for k in range(test_gt.shape[0]):
        psnr_val = torch_psnr(model_out[k, :, 0, :, :], test_gt[k, :, 0, :, :])
        ssim_val = torch_ssim(model_out[k, :, 0, :, :], test_gt[k, :, 0, :, :])
        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())
    pred = np.transpose(model_out.detach().cpu().numpy(), (0, 3, 4, 1, 2)).astype(np.float32) # bs h w c s
    truth = np.transpose(test_gt.cpu().numpy(), (0, 3, 4, 1, 2)).astype(np.float32)
    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    logger.info('testing psnr = {:.2f}, ssim = {:.3f}'
                .format(psnr_mean, ssim_mean))
    model.train()
    return pred, truth, psnr_list, ssim_list, psnr_mean, ssim_mean

def main():
    set_seed(seed)
    logger = gen_log(model_path)
    logger.info("Learning rate:{}, batch_size:{}.\n".format(opt.learning_rate, opt.batch_size))
    psnr_max = 0
    for epoch in range(1, opt.max_epoch + 1):
        train(epoch, logger)
        (pred, truth, psnr_all, ssim_all, psnr_mean, ssim_mean) = test(epoch, logger)
        scheduler.step()
        if (psnr_mean > psnr_max and psnr_mean != np.inf):
            psnr_max = psnr_mean 
            name = result_path + '/' + 'Test_{}_{:.2f}_{:.3f}'.format(epoch, psnr_max, ssim_mean) 
            # scio.savemat(name, {'truth': truth, 'pred': pred, 'psnr_list': psnr_all, 'ssim_list': ssim_all})
            draw_test_images(truth, pred, name, psnr_all, ssim_all)
            checkpoint(model, epoch, model_path, logger)

if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main()


