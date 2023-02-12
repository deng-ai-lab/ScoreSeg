import functools
import logging
import torch
from torch.nn import init
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger('base')


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(
            weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{:s}] not implemented'.format(init_type))


####################
# define network
####################


# Score-based models (Diffusion models)
def define_G(opt):
    model_opt = opt['model']
    if model_opt['which_model_G'] == 'ddpm':
        from .ddpm_modules import diffusion, unet
    elif model_opt['which_model_G'] == 'sr3':
        from .sr3_modules import diffusion, unet
    if ('norm_groups' not in model_opt['unet']) or model_opt['unet']['norm_groups'] is None:
        model_opt['unet']['norm_groups'] = 32
    model = unet.UNet(
        in_channel=model_opt['unet']['in_channel'],
        out_channel=model_opt['unet']['out_channel'],
        norm_groups=model_opt['unet']['norm_groups'],
        inner_channel=model_opt['unet']['inner_channel'],
        channel_mults=model_opt['unet']['channel_multiplier'],
        attn_res=model_opt['unet']['attn_res'],
        res_blocks=model_opt['unet']['res_blocks'],
        dropout=model_opt['unet']['dropout'],
        image_size=model_opt['diffusion']['image_size']
    )
    netG = diffusion.GaussianDiffusion(
        model,
        image_size=model_opt['diffusion']['image_size'],
        channels=model_opt['diffusion']['channels'],
        loss_type=model_opt['diffusion']['loss'],  # L1 or L2
        conditional=model_opt['diffusion']['conditional'],
        schedule_opt=model_opt['beta_schedule']['train']
    )

    if opt['phase'] == 'train':
        # init_weights(netG, init_type='kaiming', scale=0.1)
        init_weights(netG, init_type='orthogonal')

    assert torch.cuda.is_available(), 'Deformable Transformer only implemented on GPU'
    print("DDP training(define DDPM), gpus:{}".format(str(opt['local_rank'])))
    netG = DDP(netG.cuda(), device_ids=[opt['local_rank']])
    return netG


# Semantic Segmentation Network
def define_Seg(opt):
    Seg_model_opt = opt['model_Seg']
    diffusion_model_opt = opt['model']
    seg_type = opt['model_Seg']['type']
    if seg_type == 'Linear':
        from .seg_modules.seg_head_Linear_concat import seg_head_linear
        seg_head = seg_head_linear
    elif seg_type == 'Deform_concat':
        from .seg_modules.seg_head_Deform_concat import seg_head_Deform
        seg_head = seg_head_Deform
    elif seg_type == 'Deform_mean':
        from .seg_modules.seg_head_Deform_mean import seg_head_Deform
        seg_head = seg_head_Deform
    elif seg_type == 'Deform_we':
        from .seg_modules.seg_head_Deform_we import seg_head_Deform
        seg_head = seg_head_Deform
    elif seg_type == 'Deform_se':
        from .seg_modules.seg_head_Deform_se import seg_head_Deform
        seg_head = seg_head_Deform
    else:
        raise NotImplementedError('Please choose the supported segmentation head')

    params_dict = {'feat_scales': Seg_model_opt['feat_scales'],
                   'out_channels': Seg_model_opt['out_channels'],
                   'inner_channel': diffusion_model_opt['unet']['inner_channel'],
                   'channel_multiplier': diffusion_model_opt['unet']['channel_multiplier'],
                   'img_size': Seg_model_opt['output_size'],
                   'time_steps': Seg_model_opt["t"]}
    if 'Deform' in seg_type:
        params_dict['hidden_dim'] = Seg_model_opt['hidden_dim']
        params_dict['downsample_times'] = Seg_model_opt['downsample_times']

    netSeg = seg_head(**params_dict)

    if opt['phase'] == 'train':
        # Try different initialization methods
        # init_weights(netG, init_type='kaiming', scale=0.1)
        init_weights(netSeg, init_type='orthogonal')

    assert torch.cuda.is_available(), 'Deformable Transformer only implemented on GPUs'
    print("DDP training(define Segmentation modules), gpus:{}".format(str(opt['local_rank'])))
    netSeg = DDP(netSeg.cuda(), device_ids=[opt['local_rank']])

    return netSeg
