import logging
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import model.networks as networks
from .base_model import BaseModel
from utils.metric_tools import ConfuseMatrixMeter
from utils.torchutils import get_scheduler
from losses import *

logger = logging.getLogger('base')


class Seg(BaseModel):
    def __init__(self, opt):
        super(Seg, self).__init__(opt)
        # define network and load pretrained models
        self.netSeg = self.set_device(networks.define_Seg(opt))
        self.local_rank = opt['local_rank']
        self.dataset_name = opt['datasets']['train']['name']
        self.ignore_index = None
        if 'potsdam' in self.dataset_name:
            self.ignore_index = 6
        elif 'vaihingen' in self.dataset_name:
            self.ignore_index = 6
        elif 'deepglobe' in self.dataset_name:
            self.ignore_index = 6

        # set loss and load resume state
        self.loss_type = opt['model_Seg']['loss_type']
        if self.loss_type == 'JointLoss':
            self.loss_func = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=self.ignore_index),
                                       DiceLoss(smooth=0.05, ignore_index=self.ignore_index), 1.0, 1.0)
        else:
            raise NotImplementedError('choose the supported loss')

        if self.opt['phase'] == 'train':
            self.netSeg.train()
            # find the parameters to optimize
            Seg_params = list(self.netSeg.parameters())

            if opt['train']["optimizer"]["type"] == "adam":
                self.optSeg = torch.optim.Adam(
                    Seg_params, lr=opt['train']["optimizer"]["lr"])
            elif opt['train']["optimizer"]["type"] == "adamw":
                self.optSeg = torch.optim.AdamW(
                    Seg_params, lr=opt['train']["optimizer"]["lr"])
            else:
                raise NotImplementedError(
                    'Optimizer [{:s}] not implemented'.format(opt['train']["optimizer"]["type"]))
            self.log_dict = OrderedDict()

            # Define learning rate sheduler
            self.exp_lr_scheduler_netSeg = get_scheduler(optimizer=self.optSeg, args=opt['train'])
        else:
            self.netSeg.eval()
            self.log_dict = OrderedDict()

        self.print_network()
        self.load_network()

        self.running_metric = ConfuseMatrixMeter(n_class=opt['model_Seg']['out_channels']
                                                 , dataset_name=self.dataset_name)

    # Feeding all data to the Seg model
    def feed_data(self, feats, data):
        self.feats = feats
        self.data = self.set_device(data)

    # Optimize the parameters of the Seg model
    def optimize_parameters(self):
        self.optSeg.zero_grad()
        if isinstance(self.netSeg, DDP):
            self.pred = self.netSeg.module.forward(self.feats)
        else:
            self.pred = self.netSeg(self.feats)

        # coarse prediction and fine prediction
        if 'OCR' in self.opt['model_Seg']['type']:
            loss = self.opt['model_Seg']['coarse_weights'] * self.loss_func(self.pred[0],
                                                                            self.data["gt_semantic_seg"].long()) \
                   + self.loss_func(self.pred[1], self.data["gt_semantic_seg"].long())
            # only the fine prediction is used to evaluate
            self.pred = self.pred[1]
        else:
            loss = self.loss_func(self.pred, self.data["gt_semantic_seg"].long())
        dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        loss.backward()
        self.optSeg.step()
        self.log_dict['loss'] = loss.item()

    # Testing on given data
    def test(self):
        self.netSeg.eval()
        with torch.no_grad():
            if isinstance(self.netSeg, DDP):
                self.pred = self.netSeg.module.forward(self.feats)
            else:
                self.pred = self.netSeg(self.feats)
            # coarse prediction and fine prediction
            if 'OCR' in self.opt['model_Seg']['type']:
                loss = self.opt['model_Seg']['coarse_weights'] * self.loss_func(self.pred[0],
                                                                                self.data["gt_semantic_seg"].long()) \
                       + self.loss_func(self.pred[1], self.data["gt_semantic_seg"].long())
                self.pred = self.pred[1]
            else:
                loss = self.loss_func(self.pred, self.data["gt_semantic_seg"].long())
            self.log_dict['loss'] = loss.item()
        self.netSeg.train()

    # Get current log
    def get_current_log(self):
        return self.log_dict

    # Get current visuals
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['pred'] = torch.argmax(self.pred, dim=1, keepdim=False)
        out_dict['gt'] = self.data['gt_semantic_seg']
        return out_dict

    # Printing the Seg network
    def print_network(self):
        s, n = self.get_network_description(self.netSeg)
        if isinstance(self.netSeg, DDP):
            if self.local_rank != 0:
                return
            net_struc_str = '{} - {}'.format(self.netSeg.__class__.__name__,
                                             self.netSeg.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netSeg.__class__.__name__)

        logger.info(
            'Segmentation Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    # Saving the network parameters
    def save_network(self, epoch, is_best_model=False):
        Seg_gen_path = os.path.join(
            self.opt['path_Seg']['checkpoint'], 'Seg_model_E{}_gen.pth'.format(epoch))
        Seg_opt_path = os.path.join(
            self.opt['path_Seg']['checkpoint'], 'Seg_model_E{}_opt.pth'.format(epoch))

        if is_best_model:
            best_Seg_gen_path = os.path.join(
                self.opt['path_Seg']['checkpoint'], 'best_Seg_model_gen.pth')
            best_Seg_opt_path = os.path.join(
                self.opt['path_Seg']['checkpoint'], 'best_Seg_model_opt.pth')

        # Save Segmentation model parameters
        network = self.netSeg
        if isinstance(self.netSeg, DDP):
            if self.local_rank != 0:
                return
            network = network.module
            if not os.path.exists(self.opt['path_Seg']['checkpoint']):
                print('create new folder: {}'.format(self.opt['path_Seg']['checkpoint']))
                os.makedirs(self.opt['path_Seg']['checkpoint'])
        else:
            if not os.path.exists(self.opt['path_Seg']['checkpoint']):
                print('create new folder: {}'.format(self.opt['path_Seg']['checkpoint']))
                os.makedirs(self.opt['path_Seg']['checkpoint'])
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()

        # Save Segmentation optimizer parameters
        opt_state = {'epoch': epoch, 'scheduler': None, 'optimizer': self.optSeg.state_dict()}
        if is_best_model:
            torch.save(state_dict, best_Seg_gen_path)
            torch.save(opt_state, best_Seg_opt_path)
            logger.info(
                'Saved best Seg model in [{:s}] ...'.format(best_Seg_gen_path))
        else:
            torch.save(state_dict, Seg_gen_path)
            torch.save(opt_state, Seg_opt_path)
            logger.info(
                'Saved current Seg model in [{:s}] ...'.format(Seg_gen_path))

    # Resume training
    def load_network(self):
        load_path = self.opt['path_Seg']['resume_state']
        self.begin_epoch = 0
        if load_path is not None:
            logger.info(
                'Loading pretrained model for Seg model [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)

            # segmentation model
            network = self.netSeg
            if isinstance(self.netSeg, DDP):
                network = network.module

            network.load_state_dict(torch.load(
                gen_path), strict=True)

            if self.opt['phase'] == 'train':
                opt = torch.load(opt_path)
                self.optSeg.load_state_dict(opt['optimizer'])
                self.begin_epoch = opt['epoch']

    # Functions related to computing performance metrics for Semantic Segmentation
    def _update_metric(self):
        """
        update metric
        """
        G_pred = self.pred.detach()
        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(),
                                                      gt=self.data['gt_semantic_seg'].detach().cpu().numpy(),)
        return current_score

    # Collecting status of the current running batch
    def _collect_running_batch_states(self):
        self.running_acc = self._update_metric()
        self.log_dict['running_acc'] = self.running_acc.item()

    # Collect the status of the epoch
    def _collect_epoch_states(self):
        scores = self.running_metric.get_scores()
        self.epoch_acc = scores['mf1']
        self.log_dict['epoch_acc'] = self.epoch_acc.item()

        for k, v in scores.items():
            self.log_dict[k] = v
            # message += '%s: %.5f ' % (k, v)

    # Rest all the performance metrics
    def _clear_cache(self):
        self.running_metric.clear()

    # Functions related to learning rate scheduler
    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_netSeg.step()
