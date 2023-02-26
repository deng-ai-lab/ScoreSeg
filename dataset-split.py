import os
import numpy as np
import shutil
import multiprocessing as mp
import multiprocessing.pool as mpp
import argparse

'''
Given the ratio of selected images number for training
Or the images number for validating or testing
'''


def construct_fs_dataset(params):
    (source_path, target_path, log_path, phase, ratio, num) = params
    print('Mask sure your images and masks have corresponding indexex')
    if os.path.exists(target_path):
        raise NotImplementedError('target path is not None!')
    else:
        os.makedirs(target_path + '/images_256')
        os.makedirs(target_path + '/masks_256')

    source_images = os.listdir(source_path + '/images_256')
    source_masks = os.listdir(source_path + '/masks_256')
    source_images.sort()
    source_masks.sort()
    assert len(source_images) == len(source_masks), 'check the original split dataset'
    whole_num = len(source_images)
    print('The whole num: ', whole_num)
    if phase == 'train' and ratio is not None:
        num = int(ratio * whole_num)
    else:
        assert num > 0, 'check (ratio) and (num)'

    selected_index = np.random.choice(range(whole_num), num, replace=False)
    with open(log_path, 'w') as f:
        for _ in selected_index:
            selected_image_path = source_path + '/images_256/' + source_images[_]
            selected_mask_path = source_path + '/masks_256/' + source_masks[_]
            f.write(selected_image_path + '\n')
            shutil.copy(selected_image_path, target_path + '/images_256')
            shutil.copy(selected_mask_path, target_path + '/masks_256')

    print('Have constructed {} dataset'.format(phase))
    print('The selected num: {}'.format(num))
    print('The selected image index can be found in {}'.format(log_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dataset split')
    parser.add_argument('--dataset_name', type=str, choices=['potsdam', 'vaihingen', 'deepglobe'], default='potsdam')
    parser.add_argument('--phase', type=str, choices=['train', 'test'], default='train')
    args = parser.parse_args()

    if args.dataset_name == 'potsdam':
        if args.phase == 'train':
            source_path = '../data/potsdam/train'
            phase = 'train'
            num = None
            params = [(source_path,
                       '../data/potsdam/train_{}'.format(int(100 * ratio)),  # target_path
                       './indexes/potsdam_image_indexes_{}.txt'.format(int(100 * ratio)),  # log_path
                       phase, ratio, num) for ratio in [0.01, 0.05, 0.1]]
        else:
            source_path = '../data/potsdam/test'
            phase = 'test'
            ratio = None
            num = 1500
            params = [(source_path,
                       '../data/potsdam/fs_test',  # target_path
                       './indexes/potsdam_chosen_test_image_indexes.txt',  # log_path
                       phase, ratio, num)]

    elif args.dataset_name == 'vaihingen':
        if args.phase == 'train':
            source_path = '../data/vaihingen/train'
            phase = 'train'
            num = None
            params = [(source_path,
                       '../data/vaihingen/train_{}'.format(int(100 * ratio)),  # target_path
                       './indexes/vaihingen_image_indexes_{}.txt'.format(int(100 * ratio)),  # log_path
                       phase, ratio, num) for ratio in [0.01, 0.05, 0.1]]
        else:
            raise NotImplementedError('For testing, directly use the test dataset (1514 images)')

    elif args.dataset_name == 'deepglobe':
        if args.phase == 'train':
            source_path = '../data/DeepGlobe/train'
            phase = 'train'
            num = None
            params = [(source_path,
                       '../data/DeepGlobe/train_{}'.format(int(100 * ratio)),  # target_path
                       './indexes/DeepGlobe_image_indexes_{}.txt'.format(int(100 * ratio)),  # log_path
                       phase, ratio, num) for ratio in [0.01, 0.05, 0.1]]
        else:
            source_path = '../data/DeepGlobe/test'
            phase = 'test'
            ratio = None
            num = 1500
            params = [(source_path,
                       '../data/DeepGlobe/fs_test',  # target_path
                       './indexes/DeepGlobe_chosen_test_image_indexes.txt',  # log_path
                       phase, ratio, num)]

    else:
        raise NotImplementedError('Wrong parameters')

    print(params)
    mpp.Pool(processes=int(mp.cpu_count())).map(construct_fs_dataset, params)
