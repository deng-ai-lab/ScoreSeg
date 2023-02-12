import math
import cv2
import numpy as np
from torchvision.utils import make_grid


def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    """
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    """
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
             (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(
            math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()

    return img_np.astype(out_type)


def pv2rgb(mask, dataname):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    if 'potsdam' in dataname:
        mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
        mask_rgb[np.all(mask_convert == 1, axis=0)] = [0, 0, 255]
        mask_rgb[np.all(mask_convert == 2, axis=0)] = [0, 255, 255]
        mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 255, 0]
        mask_rgb[np.all(mask_convert == 4, axis=0)] = [255, 255, 0]
        mask_rgb[np.all(mask_convert == 5, axis=0)] = [255, 0, 0]
    elif 'vaihingen' in dataname:
        mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
        mask_rgb[np.all(mask_convert == 1, axis=0)] = [0, 0, 255]
        mask_rgb[np.all(mask_convert == 2, axis=0)] = [0, 255, 255]
        mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 255, 0]
        mask_rgb[np.all(mask_convert == 4, axis=0)] = [255, 255, 0]
        mask_rgb[np.all(mask_convert == 5, axis=0)] = [255, 0, 0]
    elif 'deepglobe' in dataname:
        mask_rgb[np.all(mask_convert == 0, axis=0)] = [0, 255, 255]
        mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 255, 0]
        mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 0, 255]
        mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 255, 0]
        mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 0, 255]
        mask_rgb[np.all(mask_convert == 5, axis=0)] = [255, 255, 255]
    else:
        raise NotImplementedError('refer to dataset name')
    return mask_rgb


def seg_mask2img(tensor, out_type=np.uint8, dataname='potsdam'):
    if 'potsdam' in dataname:
        CLASSES = ('ImSurf', 'Building', 'LowVeg', 'Tree', 'Car', 'Clutter')
        PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
    elif 'vaihingen' in dataname:
        CLASSES = ('ImSurf', 'Building', 'LowVeg', 'Tree', 'Car', 'Clutter')
        PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
    elif 'deepglobe' in dataname:
        CLASSES = ('urban', 'agriculture', 'rangeland', 'forest', 'water', 'barren')
        PALETTE = [[0, 255, 255], [255, 255, 0], [255, 0, 255], [0, 255, 0], [0, 0, 255], [255, 255, 255]]
    else:
        raise NotImplementedError('dataset name [{}] is not supported'.format(dataname))

    tensor = tensor.int().cpu()
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(
            math.sqrt(n_img)), normalize=False).numpy()
    else:
        raise TypeError(
            'Only support 4D tensor. But received with dimension: {:d}'.format(n_dim))

    if out_type == np.uint8:
        img_np = pv2rgb(img_np[0, :, :].copy(), dataname)
    return img_np.astype(out_type)


def save_img(img, img_path):
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

