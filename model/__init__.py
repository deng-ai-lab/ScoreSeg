import logging
logger = logging.getLogger('base')


def create_model(opt):
    from .model import ScoreModels as M
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m


def create_seg_model(opt):
    from .seg_model import Seg as M
    m = M(opt)
    logger.info('Segmentation Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
