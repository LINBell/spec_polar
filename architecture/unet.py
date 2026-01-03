import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast  # 混合精度训练

def add_gaussian_noise(measurements, noise_level=0.05):
    """
    在测量值上添加高斯白噪声
    
    Args:
        measurements: 形状为 [bs, 1, h, w] 的torch张量
        noise_level: 噪声水平，0.05 表示 5%
    
    Returns:
        添加噪声后的测量值(torch张量)
    """
    # 确保输入是torch张量
    if not torch.is_tensor(measurements):
        measurements = torch.tensor(measurements, dtype=torch.float32)
    
    # 计算测量值的标准差作为噪声的基准
    std = torch.std(measurements)
    
    # 生成与测量值相同形状的高斯噪声
    noise = torch.randn_like(measurements) * std * noise_level
    
    # 添加噪声
    noisy_measurements = measurements + noise
    
    return noisy_measurements

class Capture_Data(nn.Module):
    def __init__(self,  nl_in, need_noise=False):
        super().__init__()
        self.need_noise = need_noise
        self.nl_in = nl_in

    def forward(self, spec_data, filter):
        bs, c, s, h, w = spec_data.size()  # 9 4 4 200 200
        cap_data = torch.sum(spec_data * filter, dim=(1, 2)).unsqueeze(-1)  # 9 200 200 1
        cap_data = cap_data.permute(0, 3, 1, 2)

        cap_data = cap_data / (c*s)
        # cap_data = (cap_data - torch.min(cap_data)) / (torch.max(cap_data) - torch.min(cap_data))


        ## output---------------------------
        if self.need_noise:
            cap_data_noise = add_gaussian_noise(cap_data)
            return cap_data_noise
        return cap_data

# ====== 精确对应原1D结构的2D模型 ======
class Residual_Block(nn.Module):
    def __init__(self, channels, kernel_size):
        super(Residual_Block,self).__init__()
        self.conv1=nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=kernel_size, padding=kernel_size//2, bias=True)
        self.relu=nn.ReLU()
        # self.relu=nn.Tanh()
        self.conv2=nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=kernel_size, padding=kernel_size//2, bias=True)
        
    def forward(self,x):
        residual=x
        out=self.conv1(x)
        out=self.relu(out)
        out=self.conv2(out)
        out+=residual
        return out

    
channel_nums = 64
kernel_size = 5
    
class unet(nn.Module):
    def __init__(self,CHANNEL, channel_nums=64, kernel_size=5, raw_flag=False):
        super(unet, self).__init__()
        self.channel = CHANNEL
        self.raw_flag = raw_flag
        self.compress = Capture_Data(nl_in=0.15, need_noise=False)
        self.conv_HSI_input = nn.Conv2d(CHANNEL, channel_nums, kernel_size, padding=kernel_size//2)
        layers = []
        for i in range(10):
            layers += [Residual_Block(channel_nums, kernel_size)]
        # for i in range(9):
        #     layers += [Residual_Block(channel_nums, kernel_size)]
        # layers += [Residual_Block_tanh(channel_nums, kernel_size)]
        self.layers = torch.nn.Sequential(*layers)
        self.conv_HSI_output = nn.Conv2d(channel_nums, CHANNEL, kernel_size, padding=kernel_size//2)
            
    def forward(self, coded_patch, mask=None):
        if not self.raw_flag:
            bs, c, s, h, w = coded_patch.size()  # 8 21 4 128 128
            coded_patch = self.compress(coded_patch, mask)  # 8 1 128 128
            coded_patch = coded_patch.repeat(1, self.channel, 1, 1)  
        # print(coded_patch.shape)
        else:
            h, w = coded_patch.size()
            coded_patch = coded_patch.unsqueeze(0).unsqueeze(0)
            coded_patch = coded_patch.repeat(1, self.channel, 1, 1)
        x = self.conv_HSI_input(coded_patch)
        x = self.layers(x)
        output = self.conv_HSI_output(x)
        return output
    

    
if __name__ == "__main__":
    # 测试模型结构
    model = unet(CHANNEL=21, channel_nums=64, kernel_size=5, raw_flag=False)
    # data = torch.randn(2, 21, 4, 128, 128)  # 示例输入数据
    # mask = torch.randn(2, 21, 4, 128, 128)  #
    data = torch.randn(2, 21, 1, 280, 128)  # 示例输入数据
    mask = torch.randn(2, 21, 1, 280, 128)  #
    output = model(data, mask)
    print("输出形状:", output.shape)  