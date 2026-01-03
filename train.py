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


os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id



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
# train_set = LoadTraining(opt.data_path)
train_set = LoadTraining_npy(opt.data_path)
test_data = LoadTraining_npy(opt.test_path)

# saving path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
result_path = opt.outf + '/mstm_5by5/' + '/result/'
model_path = opt.outf + '/mstm_5by5/' + '/model/'


if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)

# model
if opt.method=='hdnet':
    model, FDL_loss = model_generator(opt.method, opt.pretrained_model_path).cuda()
else:
    model = model_generator(opt.method, opt.pretrained_model_path).cuda()

# block_path = '/data4T/lzj/mst_spectral/simulation/train_code/exp/mst_l/mstl_5by5/model/model_epoch_162.pth'
# model.load_state_dict(torch.load(block_path))


# optimizing
# optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))

curve_params = list(map(id, model.compress.parameters()))
# curve_params = list(map(id, model.module.compress.parameters()))
base_params = filter(lambda p: id(p) not in curve_params, model.parameters())



optimizer = torch.optim.Adam(
    [{'params': base_params}, {'params': model.compress.parameters(), 'lr': 10 * opt.learning_rate}],
    lr=opt.learning_rate,
    betas=(0.9, 0.999)
)


if opt.scheduler=='MultiStepLR':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
elif opt.scheduler=='CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epoch, eta_min=1e-6)
mse = torch.nn.MSELoss().cuda()
L1_loss = torch.nn.L1Loss().cuda()
L2_loss = torch.nn.MSELoss().cuda()
out_spec_channel = 21



def train(epoch, logger):

    epoch_loss = 0
    begin = time.time()
    batch_num = int(np.floor(opt.epoch_sam_num / opt.batch_size))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.ion()
    plt.show()
    x = list(range(1, opt.spec_dim + 1))

    # 使用tqdm显示进度条
    for i in tqdm(range(batch_num), desc=f'Epoch {epoch}', unit='batch'):
        gt_batch = shuffle_crop(train_set, opt.batch_size)  # 5 28 256 256
        spec_data = gt_batch.clone()
        # gt = Variable(gt_batch).cuda().float()
        # input_meas = init_meas(gt, mask3d_batch_train, opt.input_setting)   # mask 5 28 256 256   input_meas为input和mask三维相乘 压缩成一维再还原为5 28 256 256
        # input_meas = gt   # mask 5 28 256 256   input_meas为input和mask三维相乘 压缩成一维再还原为5 28 256 256
        optimizer.zero_grad()
        real_gt = spec_data
        input = spec_data / opt.ratio



        model_out = model(input, input_mask_train)
        # loss = torch.sqrt(mse(model_out, gt))

        if opt.need_disp:
            data1 = model_out.detach()[0].cpu().numpy().transpose(1, 2, 0)
            data2 = real_gt.detach()[0].cpu().numpy().transpose(1, 2, 0)
            plt.figure()
            plt.subplot(1,2,1)
            # plt.imshow(stack_21(data1) * 3)
            plt.subplot(1,2,2)
            # plt.imshow(stack_21(data2) * 5)
            plt.show(block=False)  # 非阻塞模式
            plt.pause(1.5)  # 显示 2 秒后继续
            plt.close()  # 关闭当前图形窗口
            pre_data = model_out[0].detach()
            gt_data = real_gt[0].detach()
            psnr = torch_psnr(pre_data, gt_data)
            print(psnr.item())

        pre_loss = L1_loss(model_out, real_gt)
        grad_spec_pre = model_out[:, 1:out_spec_channel, :, :] - model_out[:, 0:out_spec_channel - 1, :, :]
        grad_spec_label = real_gt[:, 1:out_spec_channel, :, :] - real_gt[:, 0:out_spec_channel - 1, :, :]
        spec_loss = L2_loss(grad_spec_pre, grad_spec_label)
        regualr_loss = 0

        curve_learned = []
        for name, param in model.compress.named_parameters():
            curve_learned.append(param)
            deg = param[:, 1:, :, :] - param[:, :out_spec_channel - 1, :, :]
            regualr_loss = regualr_loss + torch.pow((deg[:, 1:, :, :] - deg[:, :out_spec_channel - 2, :, :]), 2).sum() / (
                    out_spec_channel - 2)

        loss = pre_loss + 1e-1 * spec_loss + 1e-7 * regualr_loss

        for p in model.compress.parameters():
            p.data.clamp_(0, 1)

        if i > 0:
            ax.cla()
        flag_count = 0
        if opt.num_sp == 3:
            color_list = ['r', 'g', 'b', 'yellow', 'pink', 'black', 'chocolate', 'darkviolet', 'dimgray']
        else:
            color_list = [
                'red', 'green', 'blue', 'yellow', 'cyan',
                'Magenta', 'Black', 'White', 'Gray', 'Silver',
                'Maroon', 'Olive', 'darkviolet', 'Purple', 'Navy',
                'Orange', 'Pink', 'Lime', 'Teal', 'Indigo',
                'Coral', 'Gold', 'Brown', 'Beige', 'Turquoise'
            ]
        #
        # a = torch.squeeze(list(model.compress.parameters())[1])
        # print(a)
        # print(loss.item())

        for name, param in model.compress.named_parameters():
            flag_count = flag_count+1
            curve_response = param.clone().cpu().data.numpy().reshape((opt.spec_dim,1))
            lines = ax.plot(x, curve_response .reshape((-1)), color_list[flag_count-1], lw=1)
        plt.pause(0.05)


        epoch_loss += loss.data
        loss.backward()
        optimizer.step()



    end = time.time()

    logger.info("===> Epoch {} Complete: Avg. Loss: {:.6f} time: {:.2f}".
                format(epoch, epoch_loss / batch_num, (end - begin)))
    return 0

def test(epoch, logger):
    psnr_list, ssim_list = [], []
    test_gt = np.stack(test_data, axis=0).transpose(0, 3, 1, 2)
    test_gt = test_gt[:, :, :480, :480]

    test_gt = torch.tensor(test_gt).cuda()

    test_input = test_gt / opt.ratio

    # input_meas = init_meas(test_gt, mask3d_batch_test, opt.input_setting)
    model.eval()
    begin = time.time()
    with torch.no_grad():
        model_out = model(test_input, input_mask_test)

    end = time.time()
    for k in range(test_gt.shape[0]):
        psnr_val = torch_psnr(model_out[k, :, :, :], test_gt[k, :, :, :])
        ssim_val = torch_ssim(model_out[k, :, :, :], test_gt[k, :, :, :])
        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())
    pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    logger.info('===> Epoch {}: testing psnr = {:.2f}, ssim = {:.3f}, time: {:.2f}'
                .format(epoch, psnr_mean, ssim_mean,(end - begin)))
    model.train()
    return pred, truth, psnr_list, ssim_list, psnr_mean, ssim_mean

def main():
    logger = gen_log(model_path)
    logger.info("Learning rate:{}, batch_size:{}.\n".format(opt.learning_rate, opt.batch_size))
    psnr_max = 0
    for epoch in range(1, opt.max_epoch + 1):
        train(epoch, logger)
        (pred, truth, psnr_all, ssim_all, psnr_mean, ssim_mean) = test(epoch, logger)
        scheduler.step()
        if psnr_mean > psnr_max:
            psnr_max = psnr_mean

            name = result_path + '/' + 'Test_{}_{:.2f}_{:.3f}'.format(epoch, psnr_max, ssim_mean) + '.mat'
            scio.savemat(name, {'truth': truth, 'pred': pred, 'psnr_list': psnr_all, 'ssim_list': ssim_all})
            checkpoint(model, epoch, model_path, logger)

if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main()


