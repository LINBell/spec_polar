import argparse
import sys

import template
print("命令行参数：", sys.argv)
parser = argparse.ArgumentParser(description="HyperSpectral Image Reconstruction Toolbox")
parser.add_argument('--template', default='mst',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument("--gpu_id", type=str, default='0')

# Data specifications
parser.add_argument('--data_root', type=str, default='../../datasets/', help='dataset directory')

# Saving specifications
parser.add_argument('--outf', type=str, default='./exp/mst_l/', help='saving_path')

# Model specifications
parser.add_argument('--method', type=str, default='spec_polo', help='method name')
parser.add_argument('--pretrained_model_path', type=str, default=None, help='pretrained model directory')
parser.add_argument("--input_setting", type=str, default='H',
                    help='the input measurement of the network: H, HM or Y')
parser.add_argument("--input_mask", type=str, default='Phi',
                    help='the input mask of the network: Phi, Phi_PhiPhiT, Mask or None')  # Phi: shift_mask   Mask: mask

# Training specifications
parser.add_argument('--batch_size', type=int, default=5, help='the number of HSIs per batch')
parser.add_argument('--crop_size', type=int, default=100, help='the number of HSIs per batch')
parser.add_argument("--max_epoch", type=int, default=300, help='total epoch')
parser.add_argument("--scheduler", type=str, default='MultiStepLR', help='MultiStepLR or CosineAnnealingLR')
parser.add_argument("--milestones", type=int, default=[50,100,150,200,250], help='milestones for MultiStepLR')
parser.add_argument("--gamma", type=float, default=0.5, help='learning rate decay for MultiStepLR')
parser.add_argument("--epoch_sam_num", type=int, default=5000, help='the number of samples per epoch')
parser.add_argument("--learning_rate", type=float, default=0.0001)

parser.add_argument("--spec_dim", type=int, default=21)
parser.add_argument("--need_noise", type=str, default='False')
parser.add_argument("--need_disp", type=str, default='False')
parser.add_argument("--num_sp", type=int, default='5')
# parser.add_argument("--select_num", type=int, default='200')
parser.add_argument("--select_num", type=int, default='5')

# Noise set
parser.add_argument("--noise_level", type=float, default=0.5)
parser.add_argument("--ratio", type=float, default=0.1)

parser.add_argument('--save_name', type=str, default='spec_polo', help='method name')
parser.add_argument('--channel_index', type=int, default=0, help='channel index')
opt = parser.parse_args()

template.set_template(opt)



# dataset

# opt.data_path = "F:\spec_polo_temp"    # 原先为cave_1024_28
# opt.test_path = "F:\spec_polo_temp"    # 原先为cave_1024_28
# opt.data_path = "/data4T/lzj/311_denoise/hyperspectral/train"    # 原先为cave_1024_28
# opt.test_path = "/data4T/lzj/311_denoise/hyperspectral/test"    # 原先为cave_1024_28  3gpu
#
# opt.data_path = "/data8T/lzj/Spectral_data/denoised/hyperspectral/train"    #
# opt.test_path = "/data8T/lzj/Spectral_data/denoised/hyperspectral/test"    # 原来的好像是有噪声的

opt.data_path = "/data8T/lzj/Spectral_data/real_denoised/hyperspectral/train"
opt.test_path = "/data8T/lzj/Spectral_data/real_denoised/hyperspectral/test"    # 10013


# opt.data_path = "/data4T/lzj_data/denoise/hyperspectral/train"    # 原先为cave_1024_28
# opt.test_path = "/data4T/lzj_data/denoise/hyperspectral/test"    # 原先为cave_1024_28  2gpu
# opt.data_path = "/data4T/lzj_data/spec_polo_data"    # 原先为cave_1024_28
# opt.test_path = "/data4T/lzj_data/spec_polo_test"    # 原先为cave_1024_28  2gpu


opt.mask_path = f"{opt.data_root}/TSA_simu_data/"
# opt.test_path = f"{opt.data_root}/TSA_simu_data/Truth/"

for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False