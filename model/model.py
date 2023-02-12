import logging
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch
import os
import model.networks as networks
from .base_model import BaseModel

logger = logging.getLogger('base')


class ScoreModels(BaseModel):
    def __init__(self, opt):
        super(ScoreModels, self).__init__(opt)
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))
        self.schedule_phase = None
        self.local_rank = opt['local_rank']

        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            if opt['train']["optimizer"]["type"] == "adam":
                self.optG = torch.optim.Adam(
                    optim_params, lr=opt['train']["optimizer"]["lr"])
            elif opt['train']["optimizer"]["type"] == "adamw":
                self.optG = torch.optim.AdamW(
                    optim_params, lr=opt['train']["optimizer"]["lr"])
            else:
                raise NotImplementedError(
                    'Optimizer [{:s}] not implemented'.format(opt['train']["optimizer"]["type"]))

            self.log_dict = OrderedDict()
        self.load_network()
        self.print_network()

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optG.zero_grad()
        if isinstance(self.netG, DDP):
            l_pix = self.netG.module.forward(self.data)
        else:
            l_pix = self.netG(self.data)
        # average across multi-gpu
        b, c, h, w = self.data['img'].shape
        l_pix = l_pix.sum() / int(b * c * h * w)
        dist.all_reduce(l_pix, op=dist.ReduceOp.AVG)
        l_pix.backward()
        self.optG.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test(self, in_channels, img_size, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, DDP):
                self.sampled_img = self.netG.module.sampling_imgs(
                    in_channels, img_size, continous)
            else:
                self.sampled_img = self.netG.sampling_imgs(
                    in_channels, img_size, continous)
        self.netG.train()

    # Get feature representations for a given image
    def get_feats(self, t):
        self.netG.eval()
        img = self.data['img']
        with torch.no_grad():
            if isinstance(self.netG, DDP):
                fe_img, fd_img = self.netG.module.feats(img, t)
            else:
                fe_img, fd_img = self.netG.feats(img, t)
        self.netG.train()
        return fe_img, fd_img

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, DDP):
                self.sampled_img = self.netG.module.sample(batch_size, continous)
            else:
                self.sampled_img = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, DDP):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, DDP):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['SAM'] = self.sampled_img.detach().float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, DDP):
            # if self.local_rank != 0:
            #     return
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # generative model
        network = self.netG
        if isinstance(self.netG, DDP):
            if self.local_rank != 0:
                return
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # optimizer
        opt_state = {'epoch': epoch, 'iter': iter_step, 'scheduler': None, 'optimizer': self.optG.state_dict()}
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, DDP):
                network = network.module

            network.load_state_dict(torch.load(
                gen_path), strict=False)

            # TODO: Load optimizer is very memory-cost
            # if self.opt['phase'] == 'train':
            #     # optimizer
            #     opt = torch.load(opt_path)
            #     self.optG.load_state_dict(opt['optimizer'])
            #     self.begin_step = opt['iter']
            #     self.begin_epoch = opt['epoch']
