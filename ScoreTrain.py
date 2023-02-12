import torch
import argparse
import logging
import torch.distributed as dist
import os
import numpy as np
from tensorboardX import SummaryWriter

import data as Data
import model as Model
import utils.logger as Logger
import utils.metrics as Metrics
from utils.wandb_logger import WandbLogger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/score_pretraining.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')

    # DDP initial
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device("cuda", local_rank)
    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    # DDP
    opt['local_rank'] = local_rank
    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb

        print("Initializing wandblog.")
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            print("Creating train dataloader.")
            train_set = Data.create_image_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            print("Unconditional Sampling. No validation dataloader required.")

    logger.info('Initial Dataset Finished')

    # model
    ScoreModel = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = ScoreModel.begin_step
    current_epoch = ScoreModel.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    ScoreModel.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                ScoreModel.feed_data(train_data)
                ScoreModel.optimize_parameters()
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = ScoreModel.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    ScoreModel.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    for idx in range(0, opt['datasets']['val']['data_len'], 1):
                        ScoreModel.test(in_channels=opt['model']['unet']['in_channel'],
                                       img_size=opt['datasets']['val']['resolution'], continous=False)
                        visuals = ScoreModel.get_current_visuals()

                        sam_img = Metrics.tensor2img(visuals['SAM'])  # uint8

                        # generation
                        Metrics.save_img(
                            sam_img, '{}/sample_{}_{}.png'.format(result_path, current_step, idx))

                        tb_logger.add_image(
                            'Iter_{}'.format(current_step),
                            np.transpose(sam_img, [2, 0, 1]),
                            idx)

                        if wandb_logger:
                            wandb_logger.log_image(
                                f'validation_{idx}',
                                np.concatenate(sam_img)
                            )

                    ScoreModel.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> Sample generation completed.'.format(
                        current_epoch, current_step))

                    if wandb_logger:
                        val_step += 1

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    ScoreModel.save_network(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch - 1})

        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for idx in range(0, opt['datasets']['val']['data_len']):
            ScoreModel.test(in_channels=opt['model']['unet']['in_channel'],
                           img_size=opt['datasets']['val']['resolution'], continous=True)
            visuals = ScoreModel.get_current_visuals()

            img_mode = 'grid'
            if img_mode == 'single':

                # single img series
                sam_img = visuals['SAM']
                sample_num = sam_img.shape[0]
                for iter in range(0, sample_num):
                    Metrics.save_img(
                        Metrics.tensor2img(sam_img[iter]),
                        '{}/{}_{}_sr_{}_{}.png'.format(result_path, current_step, idx, iter, local_rank))
            else:
                # grid img
                sam_img = Metrics.tensor2img(visuals['SAM'])  # uint8
                Metrics.save_img(
                    sam_img, '{}/sampling_process_{}_{}_{}.png'.format(result_path, current_step, idx, local_rank))
                Metrics.save_img(
                    Metrics.tensor2img(visuals['SAM'][-1]),
                    '{}/sample_{}_{}_{}.png'.format(result_path, current_step, idx, local_rank))

            if wandb_logger and opt['log_eval']:
                wandb_logger.log_eval_data(sam=Metrics.tensor2img(visuals['SAM'][-1]))

        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> Sample generation completed.'.format(
            current_epoch, current_step))

        if wandb_logger:
            if opt['log_eval']:
                wandb_logger.log_eval_table()
