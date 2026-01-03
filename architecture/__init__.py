import torch
from .MST import MST
from .GAP_Net import GAP_net
from .ADMM_Net import ADMM_net
from .TSA_Net import TSA_Net
from .HDNet import HDNet, FDL
from .DGSMP import HSI_CS
from .BIRNAT import BIRNAT
from .MST_Plus_Plus import MST_Plus_Plus
from .Lambda_Net import Lambda_Net
from .CST import CST
from .DAUHST import DAUHST
from .BiSRNet import BiSRNet
from .MST_polo import MST_polo
from .MST_fisher import MST_fisher
from .MST_polo_4channel import MST_polo_4channle
from .MST_polo_six import MST_polo_six
from .SPECAT import SPECAT
from .unet import unet
from .MST_duizhao import MST_duizhao
from .MST_with_resnet import MST_with_resnet
from .unet_polar import unet_polar
# from simulation.train_code.option import opt


def model_generator(method, pretrained_model_path=None):
    if method == 'mst_s':
        model = MST(dim=21, stage=2, num_blocks=[2, 2, 2], need_noise=True, num_sp=5).cuda()
    elif method == 'mst_m':
        model = MST(dim=21, stage=2, num_blocks=[2, 4, 4], need_noise=True, num_sp=5).cuda()
    elif method == 'mst_l':
        model = MST(dim=21, stage=2, num_blocks=[4, 7, 5], need_noise=True, num_sp=5).cuda()
    elif method == 'mst_4channel':
        model = MST(dim=4, stage=2, num_blocks=[4, 7, 5], need_noise=True, num_sp=5, need_mask_atten=False).cuda()
    elif method == 'mst_21channel':
        model = MST(dim=21, stage=2, num_blocks=[4, 7, 5], need_noise=True, num_sp=5, need_mask_atten=False).cuda()
    elif method == 'spec_polo':
        model = MST_polo(dim=21*4, stage=2, num_blocks=[4, 7, 5], need_noise=True, num_sp=5, need_mask_atten=False, raw_flag=False).cuda()
    elif method == 'spec_polo_s':
        model = MST_polo(dim=21*4, stage=2, num_blocks=[2, 2, 2], need_noise=True, num_sp=5, need_mask_atten=True, raw_flag=False).cuda()
    elif method == 'spec_polo_m':
        model = MST_polo(dim=21*4, stage=2, num_blocks=[2, 4, 4], need_noise=True, num_sp=5, need_mask_atten=True, raw_flag=False).cuda()
    elif method == 'spec_polo_test':
        model = MST_polo(dim=21*4, stage=2, num_blocks=[4, 7, 5], need_noise=False, num_sp=5, need_mask_atten=False, raw_flag=True).cuda()
    elif method == 'fisher':
        model = MST_fisher(dim=21, stage=2, num_blocks=[4, 7, 5], need_noise=True, num_sp=5).cuda()
    elif method == 'spec_polo_six':
        model = MST_polo_six(dim=21*6, stage=2, num_blocks=[2, 2, 2], need_noise=True, num_sp=5, need_mask_atten=False, raw_flag=False).cuda()
    elif method == 'spec_polo_six_test':
        model = MST_polo_six(dim=21*6, stage=2, num_blocks=[2, 2, 2], need_noise=True, num_sp=5, need_mask_atten=False, raw_flag=True).cuda()
    elif method == 'SPECAT':
        model = SPECAT(dim=21*4, stage=1, num_blocks=[2, 1], attention_type='full', raw_flag=False, need_noise=True).cuda()
    elif method == 'gap_net':
        model = GAP_net().cuda()
    elif method == 'admm_net':
        model = ADMM_net().cuda()
    elif method == 'tsa_net':
        model = TSA_Net().cuda()
    elif method == 'hdnet':
        model = HDNet().cuda()
        fdl_loss = FDL(loss_weight=0.7,
             alpha=2.0,
             patch_factor=4,
             ave_spectrum=True,
             log_matrix=True,
             batch_matrix=True,
             ).cuda()
    elif method == 'dgsmp':
        model = HSI_CS(Ch=28, stages=4).cuda()
    elif method == 'birnat':
        model = BIRNAT().cuda()
    elif method == 'mst_plus_plus':
        model = MST_Plus_Plus(in_channels=28, out_channels=28, n_feat=28, stage=3).cuda()
    elif method == 'lambda_net':
        model = Lambda_Net(out_ch=28).cuda()
    elif method == 'cst_s':
        model = CST(num_blocks=[1, 1, 2], sparse=True).cuda()
    elif method == 'cst_m':
        model = CST(num_blocks=[2, 2, 2], sparse=True).cuda()
    elif method == 'cst_l':
        model = CST(num_blocks=[2, 4, 6], sparse=True).cuda()
    elif method == 'cst_l_plus':
        model = CST(num_blocks=[2, 4, 6], sparse=False).cuda()
    elif 'dauhst' in method:
        num_iterations = int(method.split('_')[1][0])
        model = DAUHST(num_iterations=num_iterations).cuda()
    elif method == 'bisrnet':
        model = BiSRNet(in_channels=28, out_channels=28, n_feat=28, stage=1, num_blocks=[1,1,1]).cuda()

    else:
        print(f'Method {method} is not defined !!!!')
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()},
                              strict=True)
    if method == 'hdnet':
        return model,fdl_loss
    return model