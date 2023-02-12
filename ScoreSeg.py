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
    parser.add_argument('-c', '--config', type=str, default='config/potsdam_segmentation.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                        help='Run either train(training + validation) or testing', default='train')
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
    Logger.setup_logger('test', opt['path']['log'], 'test', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb'] and local_rank == 0:
        import wandb

        print("Initializing wandblog.")
        wandb_logger = WandbLogger(opt)
        # Training log
        wandb.define_metric('epoch')
        wandb.define_metric('training/train_step')
        wandb.define_metric("training/*", step_metric="train_step")
        # Validation log
        wandb.define_metric('validation/val_step')
        wandb.define_metric("validation/*", step_metric="val_step")
        # Initialization
        train_step = 0
        val_step = 0
    else:
        wandb_logger = None

    # Loading datasets.
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'test':
            print("Creating [train] segmentation dataloader.")
            train_set = Data.create_seg_dataset(dataset_opt, phase)
            train_loader, train_sampler = Data.create_seg_dataloader(
                train_set, dataset_opt, phase)
            opt['len_train_dataloader'] = len(train_loader)
            print("[train] dataloader length:{}".format(len(train_loader)))

        elif phase == 'val' and args.phase != 'test':
            print("Creating [val] segmentation dataloader.")
            val_set = Data.create_seg_dataset(dataset_opt, phase)
            val_loader, val_sampler = Data.create_seg_dataloader(
                val_set, dataset_opt, phase)
            opt['len_val_dataloader'] = len(val_loader)
            print("[val] dataloader length:{}".format(len(val_loader)))

        elif phase == 'test' and args.phase == 'test':
            print("Creating [test] segmentation dataloader.")
            print(phase)
            test_set = Data.create_seg_dataset(dataset_opt, phase)
            test_loader, test_sampler = Data.create_seg_dataloader(
                test_set, dataset_opt, phase)
            opt['len_test_dataloader'] = len(test_loader)
            print("[test] dataloader length:{}".format(len(test_loader)))

    # results = dict(img_id=img_id, img=img, gt_semantic_seg=mask)
    logger.info('Initial Dataset Finished')

    # Loading score-based model [also called Denoising Diffusion Probabilistic Models (DDPM)]
    ScoreModel = Model.create_model(opt)
    logger.info('Initial Diffusion Model Finished')

    # Set noise schedule for the score-based model
    ScoreModel.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    # Creating semantic segmentation model
    segmentation_model = Model.create_seg_model(opt)
    #################
    # Training loop #
    #################
    n_epoch = opt['train']['n_epoch']
    best_OA = 0.0
    start_epoch = segmentation_model.begin_epoch

    if opt['phase'] == 'train':
        assert opt['train']['save_freq'] > 0, 'check configs for saving frequency'
        for current_epoch in range(start_epoch, n_epoch):

            train_sampler.set_epoch(current_epoch)
            segmentation_model._clear_cache()
            train_result_path = '{}/train/{}'.format(opt['path_Seg']
                                                     ['results'], current_epoch)
            os.makedirs(train_result_path, exist_ok=True)

            ################
            ### training ###
            ################
            message = 'lr: %0.7f\n \n' % segmentation_model.optSeg.param_groups[0]['lr']
            logger.info(message)

            for current_step, train_data in enumerate(train_loader):
                # Feeding RSIs to score-based model and get features
                ScoreModel.feed_data(train_data)
                f_img = []
                for t in opt['model_Seg']['t']:
                    fe_img_t, fd_img_t = ScoreModel.get_feats(t=t)
                    if opt['model_Seg']['feat_type'] == "dec":
                        f_img.append(fd_img_t)
                        del fe_img_t
                    else:
                        f_img.append(fe_img_t)
                        del fd_img_t

                # Feeding features from the score-based model to the Seg model
                segmentation_model.feed_data(f_img, train_data)
                segmentation_model.optimize_parameters()
                segmentation_model._collect_running_batch_states()

                # log running batch status
                if current_step % opt['train']['train_print_freq'] == 0:
                    # message
                    logs = segmentation_model.get_current_log()
                    message = '[Training Seg]. epoch: [%d/%d]. Iter: [%d/%d], loss: %.5f, running_mf1: %.5f\n' % \
                              (current_epoch, n_epoch - 1, current_step, len(train_loader), logs['loss'],
                               logs['running_acc'])
                    logger.info(message)

                    # visualization
                    visuals = segmentation_model.get_current_visuals()
                    assert train_data['gt_semantic_seg'].shape == visuals['pred'].shape, 'mask shape: {} != pred ' \
                                                                                         'shape: {}'.format(
                        train_data['gt_semantic_seg'].shape, visuals['pred'].shape)

                    img = Metrics.tensor2img(train_data['img'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                    label = Metrics.seg_mask2img(train_data['gt_semantic_seg'].unsqueeze(1),  # b x 1 x h x w
                                                 out_type=np.uint8,
                                                 dataname=opt['datasets']['train']['name'])  # uint8
                    pred = Metrics.seg_mask2img(visuals['pred'].unsqueeze(1),
                                                out_type=np.uint8,
                                                dataname=opt['datasets']['train']['name'])  # uint8

                    # save imgs
                    Metrics.save_img(
                        img,
                        '{}/img_e{}_s{}_r{}.png'.format(train_result_path, current_epoch, current_step, local_rank))
                    Metrics.save_img(
                        pred,
                        '{}/pred_e{}_s{}_r{}.png'.format(train_result_path, current_epoch, current_step, local_rank))
                    Metrics.save_img(
                        label,
                        '{}/gt_e{}_s{}_r{}.png'.format(train_result_path, current_epoch, current_step, local_rank))

            ### log epoch status ###
            segmentation_model._collect_epoch_states()
            logs = segmentation_model.get_current_log()
            message = '[Training Seg (epoch summary)]: epoch: [%d/%d]. epoch_mF1=%.5f \n' % \
                      (current_epoch, n_epoch - 1, logs['epoch_acc'])
            for k, v in logs.items():
                message += '{:s}: {:.4e} '.format(k, v)
                tb_logger.add_scalar(k, v, current_step)
            message += '\n'
            logger.info(message)

            if wandb_logger:
                wandb_logger.log_metrics({
                    'training/OA': logs['OA'],
                    'training/mF1': logs['epoch_acc'],
                    'training/mIoU': logs['miou'],
                    'training/kappa': logs['kappa'],
                    'training/train_step': current_epoch
                })

            segmentation_model._clear_cache()
            segmentation_model._update_lr_schedulers()

            ##################
            ### validation ###
            ##################
            if current_epoch % opt['train']['val_freq'] == 0:
                val_sampler.set_epoch(current_epoch)
                val_result_path = '{}/val/{}'.format(opt['path_Seg']
                                                     ['results'], current_epoch)
                os.makedirs(val_result_path, exist_ok=True)

                for current_step, val_data in enumerate(val_loader):
                    # Feed data to score-based model

                    ScoreModel.feed_data(val_data)
                    f_img = []
                    for t in opt['model_Seg']['t']:
                        fe_img_t, fd_img_t = ScoreModel.get_feats(t=t)
                        if opt['model_Seg']['feat_type'] == "dec":
                            f_img.append(fd_img_t)
                            del fe_img_t
                        else:
                            f_img.append(fe_img_t)
                            del fd_img_t

                    # Feeding features from the score-based model to the Seg model
                    segmentation_model.feed_data(f_img, val_data)
                    segmentation_model.test()
                    segmentation_model._collect_running_batch_states()

                    # log running batch status for val data
                    if current_step % opt['train']['val_print_freq'] == 0:
                        # message
                        logs = segmentation_model.get_current_log()
                        message = '[Validation Seg]. epoch: [%d/%d]. Iter: [%d/%d], running_mf1: %.5f\n' % \
                                  (current_epoch, n_epoch - 1, current_step, len(val_loader), logs['running_acc'])
                        logger.info(message)

                        # visuals
                        visuals = segmentation_model.get_current_visuals()

                        img = Metrics.tensor2img(val_data['img'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                        label = Metrics.seg_mask2img(val_data['gt_semantic_seg'].unsqueeze(1),
                                                     out_type=np.uint8,
                                                     dataname=opt['datasets']['val']['name'])  # uint8
                        pred = Metrics.seg_mask2img(visuals['pred'].unsqueeze(1),
                                                    out_type=np.uint8,
                                                    dataname=opt['datasets']['val']['name'])  # uint8

                        # save imgs
                        Metrics.save_img(
                            img,
                            '{}/img_e{}_s{}_r{}.png'.format(val_result_path, current_epoch, current_step, local_rank))
                        Metrics.save_img(
                            pred,
                            '{}/pred_e{}_s{}_r{}.png'.format(val_result_path, current_epoch, current_step, local_rank))
                        Metrics.save_img(
                            label,
                            '{}/gt_e{}_s{}_r{}.png'.format(val_result_path, current_epoch, current_step, local_rank))

                segmentation_model._collect_epoch_states()
                logs = segmentation_model.get_current_log()
                message = '[Validation Seg (epoch summary)]: epoch: [%d/%d]. epoch_mF1=%.5f \n' % \
                          (current_epoch, n_epoch - 1, logs['epoch_acc'])
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    tb_logger.add_scalar(k, v, current_step)
                message += '\n'
                logger.info(message)

                if wandb_logger:
                    wandb_logger.log_metrics({
                        'validation/OA': logs['OA'],
                        'validation/mF1': logs['epoch_acc'],
                        'validation/mIoU': logs['miou'],
                        'validation/kappa': logs['kappa'],
                        'validation/val_step': current_epoch
                    })

                if logs['OA'] > best_OA:
                    is_best_model = True
                    best_OA = logs['OA']
                    logger.info(
                        '[Validation Seg] Best model updated. Saving the best model and training states.')
                    segmentation_model.save_network(current_epoch, is_best_model=is_best_model)
                if current_epoch % opt['train']['save_freq'] == 0:
                    is_best_model = False
                    logger.info('[Validation Seg]Saving the current Seg model and training states.')
                    segmentation_model.save_network(current_epoch, is_best_model=is_best_model)
                logger.info('--- Proceed To The Next Epoch ----\n \n')

                segmentation_model._clear_cache()

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch - 1})

        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation (testing).')
        test_result_path = '{}/test/'.format(opt['path_Seg']
                                             ['results'])
        os.makedirs(test_result_path, exist_ok=True)
        logger_test = logging.getLogger('test')  # test logger
        segmentation_model._clear_cache()
        for current_step, test_data in enumerate(test_loader):
            # Feed data to score-based model
            ScoreModel.feed_data(test_data)

            f_img = []
            for t in opt['model_Seg']['t']:
                fe_img_t, fd_img_t = ScoreModel.get_feats(t=t)
                if opt['model_Seg']['feat_type'] == "dec":
                    f_img.append(fd_img_t)
                    del fe_img_t
                else:
                    f_img.append(fe_img_t)
                    del fd_img_t

            # Feeding features from the score-based model to the Seg model
            segmentation_model.feed_data(f_img, test_data)
            segmentation_model.test()
            segmentation_model._collect_running_batch_states()
            # Logs
            logs = segmentation_model.get_current_log()
            message = '[Testing Seg]. Iter: [%d/%d], running_mf1: %.5f\n' % \
                      (current_step, len(test_loader), logs['running_acc'])
            logger_test.info(message)

            # visuals
            visuals = segmentation_model.get_current_visuals()

            img = Metrics.tensor2img(test_data['img'], out_type=np.uint8, min_max=(-1, 1))  # uint8
            label = Metrics.seg_mask2img(test_data['gt_semantic_seg'].unsqueeze(1),
                                         out_type=np.uint8,
                                         dataname=opt['datasets']['test']['name'])  # uint8
            pred = Metrics.seg_mask2img(visuals['pred'].unsqueeze(1),
                                        out_type=np.uint8,
                                        dataname=opt['datasets']['test']['name'])  # uint8

            # set batchsize == 1, use the image name to save prediction and groundtruth
            if len(test_data['img_id']) == 1:
                img_name = os.path.split(test_data['img_id'][0])[-1]
                Metrics.save_img(
                    img, '{}/{}_img.png'.format(test_result_path, img_name))
                Metrics.save_img(
                    pred, '{}/{}_pred.png'.format(test_result_path, img_name))
                Metrics.save_img(
                    label, '{}/{}_gt.png'.format(test_result_path, img_name))
            else:
                Metrics.save_img(
                    img, '{}/img_s{}_r{}.png'.format(test_result_path, current_step, local_rank))
                Metrics.save_img(
                    pred, '{}/pred_s{}_r{}.png'.format(test_result_path, current_step, local_rank))
                Metrics.save_img(
                    label, '{}/gt_s{}_r{}.png'.format(test_result_path, current_step, local_rank))

        segmentation_model._collect_epoch_states()
        logs = segmentation_model.get_current_log()
        message = '[Test Seg summary]: Test mF1=%.5f \n' % \
                  (logs['epoch_acc'])
        for k, v in logs.items():
            message += '{:s}: {:.4e} '.format(k, v)
            message += '\n'
        logger_test.info(message)

        if wandb_logger:
            wandb_logger.log_metrics({
                'test/OA': logs['OA'],
                'test/mF1': logs['epoch_acc'],
                'test/mIoU': logs['miou'],
                'test/kappa': logs['kappa'],
            })

        logger.info('End of testing...')
