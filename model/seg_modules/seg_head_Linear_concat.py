import torch
import torch.nn as nn
import torch.nn.functional as F


def get_in_channels(feat_scales, inner_channel, channel_multiplier):
    '''
    Get the number of input layers to the segmentation head.
    '''
    in_channels = 0
    for scale in feat_scales:
        if scale < 3:  # 256 x 256
            in_channels += inner_channel * channel_multiplier[0]
        elif scale < 6:  # 128 x 128
            in_channels += inner_channel * channel_multiplier[1]
        elif scale < 9:  # 64 x 64
            in_channels += inner_channel * channel_multiplier[2]
        elif scale < 12:  # 32 x 32
            in_channels += inner_channel * channel_multiplier[3]
        elif scale < 15:  # 16 x 16
            in_channels += inner_channel * channel_multiplier[4]
        else:
            print('Unbounded number {} for feat_scales. 0<=feat_scales<=14'.format(scale))
    return in_channels


def get_in_size(feat_scales, downsample_times=0, mode='side'):
    '''
    Get the size of input tensors to the segmentation head.
    mode: return h Or h*h
    '''
    in_size = 0
    for scale in feat_scales:
        if scale < 3:  # 256 x 256
            in_size += 256 if mode == 'side' else 256 * 256
        elif scale < 6:  # 128 x 128
            in_size += 128 if mode == 'side' else 128 * 128
        elif scale < 9:  # 64 x 64
            in_size += 64 if mode == 'side' else 64 * 64
        elif scale < 12:  # 32 x 32
            in_size += 32 if mode == 'side' else 32 * 32
        elif scale < 15:  # 16 x 16
            in_size += 16 if mode == 'side' else 16 * 16
        else:
            print('Unbounded number {} for feat_scales. 0<=feat_scales<=14'.format(scale))
    in_size = round(in_size / 2 ** downsample_times) if mode == 'side' else round(in_size / 4 ** downsample_times)
    return in_size


class seg_head_linear(nn.Module):

    def __init__(self, feat_scales, out_channels, inner_channel=None, channel_multiplier=None, img_size=256,
                 time_steps=None):
        super(seg_head_linear, self).__init__()

        self.feat_scales = feat_scales
        self.img_size = img_size
        self.time_steps = time_steps
        input_dim = 0
        for i in range(len(self.feat_scales)):
            input_dim += get_in_channels([self.feat_scales[i]], inner_channel, channel_multiplier) * len(time_steps)

        # pixel-wise classifier
        self.pixel_clfr = nn.Sequential(
            nn.Conv2d(input_dim, 128, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 32, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, out_channels, (1, 1)),
        )

    def forward(self, feats):
        # Upsample and concat
        for lvl in range(len(self.feat_scales)):
            # squeeze time dim to Channel dim (cat) [t,b,c,h,w] -> [b,c*t,h,w]
            all_time_feats = feats[0][self.feat_scales[lvl]]
            for i in range(1, len(self.time_steps)):
                all_time_feats = torch.cat((all_time_feats, feats[i][self.feat_scales[lvl]]), dim=1)
            if lvl == 0:
                all_features = F.interpolate(all_time_feats, size=(self.img_size, self.img_size), mode="bilinear")
            else:
                # concat on channel dim
                all_features = torch.cat((all_features,
                                          F.interpolate(all_time_feats, size=(self.img_size, self.img_size),
                                                        mode="bilinear")), dim=1)

        # Classifier
        pred = self.pixel_clfr(all_features)

        return pred
