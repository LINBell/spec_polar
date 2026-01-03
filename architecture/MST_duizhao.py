from functools import total_ordering
from typing import Any
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
# from simulation.train_code.utils import *
from matplotlib import pyplot as plt

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



def stack_21(data):
    data1 = data[:, :, 0]
    data2 = data[:, :, 10]
    data3 = data[:, :, 20]
    return np.stack([data1, data2, data3], axis=-1)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')



"""
input 5 28 256 256
output 5 1 256 256 -----> 5 28 256 256

"""

class My_addnoise_function_realprocess_fisher_version(torch.autograd.Function):

    @staticmethod
    def forward(ctx, spec, nl_in):
        # print('average num of photons of input = ', spec.mean())
        BATCH_SIZE, HEIGHT, WIDTH, channel_num = spec.size()

        Poisson_Tensor = torch.Tensor([20]).reshape([1, 1, 1, 1]).cuda().repeat(BATCH_SIZE, HEIGHT, WIDTH, channel_num)
        Dark_Tensor = torch.Tensor([1]).reshape([1, 1, 1, 1]).cuda().repeat(BATCH_SIZE, HEIGHT, WIDTH, channel_num)
        Gaussian_Tensor = torch.Tensor([1]).reshape([1, 1, 1, 1]).cuda().repeat(BATCH_SIZE, HEIGHT, WIDTH, channel_num)
        cons = torch.Tensor([0.0000000001]).reshape([1, 1, 1, 1]).cuda().repeat(BATCH_SIZE, HEIGHT, WIDTH, channel_num)

        Poisson_Tensor = Poisson_Tensor * nl_in
        Dark_Tensor = Dark_Tensor * nl_in
        Gaussian_Tensor = Gaussian_Tensor * nl_in

        # add poisson (shot)
        peak = spec + cons  ## num of photons
        pnoisy = torch.distributions.poisson.Poisson(rate=peak).sample()
        # add dark
        dnoisy = torch.distributions.poisson.Poisson(rate=Dark_Tensor).sample()
        # add gaussian
        gnoisy = torch.distributions.normal.Normal(loc=0, scale=Gaussian_Tensor).sample()
        # add width
        # k_noisy = torch.distributions.normal.Normal(loc = 1, scale = Wi_Tensor).sample()
        # two noise
        noisy = (pnoisy + dnoisy + gnoisy) * Poisson_Tensor / 255  # * #* k_noisy.reshape([BATCH_SIZE, HEIGHT, 1, channel_num]).repeat(1, 1, WIDTH, 1)

        # Save tensors for backward pass
        ctx.save_for_backward(spec, Poisson_Tensor, Dark_Tensor, Gaussian_Tensor, peak)

        # Return noisy image along with peak, Dark_Tensor, and Gaussian_Tensor
        # return noisy, peak, Dark_Tensor, Gaussian_Tensor ** 2
        return noisy

    @staticmethod
    def backward(ctx, grad_output, grad_peak, grad_Dark_Tensor, grad_Gaussian_Tensor):
        # Retrieve saved tensors
        # spec, Poisson_Tensor, Dark_Tensor, Gaussian_Tensor, peak = ctx.saved_tensors

        # Compute gradients
        grad_x = grad_output  # Only the gradient w.r.t. noisy is needed
        return grad_x, None



class Capture_Data(nn.Module):
    def __init__(self,  nl_in, need_noise=False):
        super().__init__()
        self.need_noise = need_noise
        self.nl_in = nl_in

    def forward(self, spec_data, filter):
        bs, c, s, h, w = spec_data.size()  # 9 4 4 200 200
        cap_data = torch.sum(spec_data * filter, dim=(1, 2)).unsqueeze(-1)  # 9 200 200 1
        cap_data = cap_data.permute(0, 3, 1, 2)

        cap_data = cap_data / (c * 2)
        # cap_data = (cap_data - torch.min(cap_data)) / (torch.max(cap_data) - torch.min(cap_data))


        ## output---------------------------
        if self.need_noise:
            cap_data_noise = add_gaussian_noise(cap_data)
            return cap_data_noise.permute(0, 2, 3, 1)
        return cap_data.permute(0, 2, 3, 1)



class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

def conv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)

def shift_back(inputs,step=2):          # input [bs,28,256,310]  output [bs, 28, 256, 256]
    [bs, nC, row, col] = inputs.shape
    down_sample = 256//row
    step = float(step)/float(down_sample*down_sample)
    out_col = row
    for i in range(nC):
        inputs[:,i,:,:out_col] = \
            inputs[:,i,:,int(step*i):int(step*i)+out_col]
    return inputs[:, :, :, :out_col]

class MaskGuidedMechanism(nn.Module):
    def __init__(
            self, n_feat):
        super(MaskGuidedMechanism, self).__init__()

        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(n_feat, n_feat, kernel_size=5, padding=2, bias=True, groups=n_feat)

    def forward(self, mask_shift):
        # x: b,c,h,w
        [bs, nC, row, col] = mask_shift.shape
        mask_shift = self.conv1(mask_shift)
        attn_map = torch.sigmoid(self.depth_conv(self.conv2(mask_shift)))
        res = mask_shift * attn_map
        mask_shift = res + mask_shift
        mask_emb = shift_back(mask_shift)
        return mask_emb

class MA(nn.Module):              # Mask attention for optical filter-based HSI system
    def __init__(
            self, n_feat):
        super(MA, self).__init__()
        self.depth_conv = nn.Conv2d(n_feat, n_feat, kernel_size=5, padding=2, bias=True, groups=n_feat)

    def forward(self, mask_3d):
        attn_map = torch.sigmoid(self.depth_conv(mask_3d))
        res =  mask_3d * attn_map
        mask_attn = res + mask_3d
        return mask_attn


class MS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            need_mask_atten = False
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        # self.mm = MA(dim)
        self.dim = dim
        self.need_mask_atten = need_mask_atten
        self.ma = MA(dim)

    def forward(self, x_in, mask=None):
        """
        x_in: [b,h,w,c]
        mask: [1,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b,h*w,c)
        q_inp = self.to_q(x)    # 5 65535 28
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        if self.need_mask_atten:  # mask 5 100 100 84
            mask_attn = self.ma(mask.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # 5 100 100 84
            # if b != 0:
            #     mask_attn = (mask_attn[0, :, :, :]).expand([b, h, w, c])
            q, k, v, mask_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                    (q_inp, k_inp, v_inp, mask_attn.flatten(1, 2)))   # 5 1 65536 28
            v = v * mask_attn
        else:
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                    (q_inp, k_inp, v_inp))   # 5 1 65536 28


        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)   # 5 1 28 28
        x = attn @ v   # b,heads,d,hw      5 1 28 65536
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)  # 5 65536 28   dim_head=28 num_head=1
        out_c = self.proj(x).view(b, h, w, c)   # 5 256 256 28
        out_p = self.pos_emb(v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)

class MSAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2,
            need_mask_atten=False
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MS_MSA(dim=dim, dim_head=dim_head, heads=heads, need_mask_atten=need_mask_atten),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, mask):
        # def forward(self, x, mask=None):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x, mask=mask.permute(0, 2, 3, 1)) + x
            # x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)  # 5 28 256 256
        return out


class MST_duizhao(nn.Module):
    def __init__(self, dim=21, stage=3, num_blocks=[2,2,2], need_noise=False, num_sp=5, need_mask_atten=False, raw_flag=False):
        super(MST_duizhao, self).__init__()
        self.dim = dim
        self.stage = stage
        self.need_noise = need_noise
        self.num_sp = num_sp
        self.need_mask_atten = need_mask_atten
        self.raw_flag = raw_flag
        # Input projection
        self.embedding = nn.Conv2d(self.dim, self.dim, 3, 1, 1, bias=False)

        self.compress = Capture_Data(nl_in=0.15, need_noise=self.need_noise)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                MSAB(
                    dim=dim_stage, num_blocks=num_blocks[i], dim_head=dim, heads=dim_stage // dim, need_mask_atten=need_mask_atten),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False)
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = MSAB(
            dim=dim_stage, dim_head=dim, heads=dim_stage // dim, num_blocks=num_blocks[-1], need_mask_atten=need_mask_atten)

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                MSAB(
                    dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i], dim_head=dim,
                    heads=(dim_stage // 2) // dim, need_mask_atten=need_mask_atten),
            ]))
            dim_stage //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, self.dim, 3, 1, 1, bias=False)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, filter_matrix):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        if not self.raw_flag:
            b, c, s, h, w = filter_matrix.size()
            # if mask == None:
            #     mask = torch.zeros((1,28,256,310)).cuda()

            x = self.compress(x, filter_matrix)
            x = x.permute(0,3,1,2).repeat_interleave(21*4, dim=1)
        else:
            h, w = x.shape
            b, c, s = 1, 21, 4
            x = x.unsqueeze(0).unsqueeze(0).repeat_interleave(21*4, dim=1)
            
        mask = filter_matrix.view(b, c*s, h, w)
        # Embedding
        fea = self.lrelu(self.embedding(x))  # 5 28 256 256

        # Encoder
        fea_encoder = []
        masks = []
        for (MSAB, FeaDownSample, MaskDownSample) in self.encoder_layers:
            fea = MSAB(fea, mask)   # 5 28 256 256
            # fea = MSAB(fea)   # 5 28 256 256
            masks.append(mask)      # 5 28 256 310
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            mask = MaskDownSample(mask)

        # Bottleneck
        fea = self.bottleneck(fea, mask)   #in and out: 5 112 64 64
        # fea = self.bottleneck(fea)   #in and out: 5 112 64 64

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)   # 5 56 128 128
            fea = Fution(torch.cat([fea, fea_encoder[self.stage-1-i]], dim=1))
            mask = masks[self.stage - 1 - i]
            fea = LeWinBlcok(fea, mask)
            # fea = LeWinBlcok(fea)

        # Mapping
        out = self.mapping(fea) + x   # 5 28 256 256

        return out

if __name__ == '__main__':
    model = MST_polo(dim=21, stage=2, num_blocks=[2, 2], need_noise=True, num_sp=5, need_mask_atten=True, raw_flag=False).cuda()
    dummy_input = torch.randn(2, 21, 100, 100).cuda()
    dummy_filter = torch.randn(2, 21, 100, 100).cuda()
    output = model(dummy_input, dummy_filter)
    print(output.shape)



















