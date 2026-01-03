from torch import nn as nn
from tqdm import tqdm
from matplotlib import pyplot as plt
from architecture import *
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
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

"""
row图像 [100,100] Capdata后 归一化后 应该直接送入无 cap过程
"""

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

seed = 529



pretrain_model_path = '/data8T/lzj/MST_Spectral_remote/simulation/train_code/exp/mst_l/tio2_10_10_4_states_test_fdtd/model/model_epoch_98.pth'
filter_path = '/data8T/lzj/MST_Spectral_remote/simulation/train_code/mask/tio2_100_mutual_10_10.npy'
raw_path = '/data8T/lzj/Spectral_data/denoised/hyperspectral/test/0201.npy'
out_path = '/data8T/lzj/MST_Spectral_remote/simulation/train_code/raw_output_test.npy'

model = model_generator('spec_polo', opt.pretrained_model_path).cuda()
model.load_state_dict(torch.load(pretrain_model_path))
filter = np.load(filter_path)
print(f'Now using {filter_path} ')
filter_matrix_test = torch.from_numpy(filter).unsqueeze(0).repeat(1, 1, 1, 1, 1).cuda().float()
filter_matrix_test = filter_matrix_test[:, :, :, :opt.crop_size, :opt.crop_size]

raw_data = torch.from_numpy(np.load(raw_path)).to(torch.float32).cuda()
raw_data = normalize_stokes(raw_data)
print(raw_data.shape)
_, _, num = raw_data.shape
def test():
    out_put = []
    set_seed(seed)
    model.eval()
    with torch.no_grad():
        for index in range(num):
            input = raw_data[:, :, index]
            print(input.shape)
            model_out = model(input, filter_matrix_test)
            model_out = model_out.view(21, 4, opt.crop_size, opt.crop_size)
    
            out_put.append(model_out.detach().cpu().numpy())  # 1 c s h w
    return out_put

if __name__ == '__main__':
    out_put = test()
    out_put = np.stack(out_put, axis=0)
    print(out_put.shape)
    np.save(out_path, out_put)  # 21 21 4 100 100
