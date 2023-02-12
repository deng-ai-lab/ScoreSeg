"""create dataset and dataloader"""
import logging
import torch.utils.data


# Create image dataset
def create_image_dataset(dataset_opt, phase):
    """create dataset"""
    mode = dataset_opt['mode']
    from data.ImageDataset import ImageDataset as Dataset
    dataset = Dataset(dataroot=dataset_opt['dataroot'],
                      resolution=dataset_opt['resolution'],
                      split=phase,
                      data_len=dataset_opt['data_len']
                      )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset


def create_dataloader(dataset, dataset_opt, phase):
    """create dataloader """
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            pin_memory=True)
    elif phase == 'val':
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))


# Create segmentation dataset
def create_seg_dataset(dataset_opt, phase):
    """create dataset"""
    if dataset_opt['name'] == 'potsdam':
        from data.potsdam_dataset import PotsdamDataset as Dataset
        from data.potsdam_dataset import train_aug, val_aug
    elif dataset_opt['name'] == 'vaihingen':
        from data.vaihingen_dataset import VaihingenDataset as Dataset
        from data.vaihingen_dataset import train_aug, val_aug
    elif dataset_opt['name'] == 'deepglobe':
        from data.deepglobe_dataset import DeepGlobeDataset as Dataset
        from data.deepglobe_dataset import train_aug, val_aug
    else:
        raise NotImplementedError('please choose the supported dataset')
    dataset = Dataset(data_root=dataset_opt['dataroot'],
                      mode=phase,
                      img_dir=dataset_opt['img_dir'],
                      mask_dir=dataset_opt['mask_dir'],
                      mosaic_ratio=dataset_opt['mosaic_ratio'],
                      transform=train_aug if phase == 'train' else val_aug,
                      )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                                  dataset_opt['name'],
                                                                  phase))
    return dataset


# Create segmentation dataloader
def create_seg_dataloader(dataset, dataset_opt, phase):
    """create dataloader """

    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=dataset_opt['use_shuffle'],
                                                              drop_last=True if phase == 'train' else False)
    if phase == 'train' or 'val' or 'test':
        # return sampler for DDP shuffle
        return (torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=False if sampler else dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            pin_memory=True,
            sampler=sampler)
            , sampler)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))
