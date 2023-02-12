from typing import List
import torch.nn.functional as F

from .deformable_attn import DeformableTransformer
from .position_encoding import *
from .seg_head_Linear_concat import get_in_channels, get_in_size

BatchNorm2d = torch.nn.SyncBatchNorm


class seg_head_Deform(nn.Module):

    def __init__(self, feat_scales, out_channels, inner_channel=None, channel_multiplier=None, img_size=256,
                 time_steps=None, hidden_dim=256, downsample_times=0):
        super(seg_head_Deform, self).__init__()

        self.hidden_dim = hidden_dim
        self.feat_scales = feat_scales  # 5 layers
        self.in_channels = get_in_channels(feat_scales, inner_channel, channel_multiplier)
        self.img_size = img_size
        self.time_steps = time_steps

        # transformers
        self.position_embedding = PositionEmbeddingSine(self.hidden_dim // 2, normalize=True)

        # Downsample layers before attention (optional)
        # Balance computation cost and results accuracy
        self.downsample_times = downsample_times
        self.downsample_layers = nn.ModuleList()
        self.input_proj = nn.ModuleList()
        if self.downsample_times == 0:
            for i in range(len(self.feat_scales)):
                in_channels = get_in_channels([self.feat_scales[i]], inner_channel, channel_multiplier)
                self.downsample_layers.append(
                    nn.Identity()
                )

                self.input_proj.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels * len(time_steps), self.hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, self.hidden_dim),
                    )
                )
        elif self.downsample_times == 1:
            for i in range(len(self.feat_scales)):
                in_channels = get_in_channels([self.feat_scales[i]], inner_channel, channel_multiplier)
                self.downsample_layers.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels * len(time_steps), self.hidden_dim, kernel_size=1, bias=False),
                        BatchNorm2d(self.hidden_dim),
                        nn.ReLU(inplace=True)
                    )
                )

                self.input_proj.append(
                    nn.Sequential(
                        nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, self.hidden_dim),
                    )
                )
        elif self.downsample_times == 2:
            for i in range(len(self.feat_scales)):
                in_channels = get_in_channels([self.feat_scales[i]], inner_channel, channel_multiplier)
                self.downsample_layers.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels * len(time_steps), self.hidden_dim, kernel_size=3, stride=2, padding=1,
                                  bias=False),
                        BatchNorm2d(self.hidden_dim),
                        nn.ReLU(inplace=True)
                    )
                )

                self.input_proj.append(
                    nn.Sequential(
                        nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, self.hidden_dim),
                    )
                )
        else:
            raise NotImplementedError('Supported parameter [downsample_times]: 0/1/2')

        self.transformer = DeformableTransformer(d_model=self.hidden_dim, nhead=8,
                                                 num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                                                 activation="relu",
                                                 num_feature_levels=len(self.feat_scales), enc_n_points=4)

        # Final classification head
        clfr_emb_dim = 64
        self.clfr_stg1 = nn.Conv2d(self.hidden_dim, clfr_emb_dim, kernel_size=3, padding=1)
        self.clfr_stg2 = nn.Conv2d(clfr_emb_dim, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, feats):
        # Decoder
        out: List[NestedTensor] = []
        pos = []
        srcs = []
        masks = []

        for lvl in range(len(self.feat_scales)):
            # squeeze time dim to Channel dim (cat) [t,b,c,h,w] -> [b,c*t,h,w]
            all_time_feats = feats[0][self.feat_scales[lvl]]
            for i in range(1, len(self.time_steps)):
                all_time_feats = torch.cat((all_time_feats, feats[i][self.feat_scales[lvl]]), dim=1)

            all_time_feats = self.input_proj[lvl](self.downsample_layers[lvl](all_time_feats))
            b, _, h, w = all_time_feats.shape
            # no padding, True for padding elements, False for non-padding elements
            mask = torch.zeros([b, h, w], dtype=torch.bool, device=all_time_feats.device)
            out.append(NestedTensor(all_time_feats, mask))
            masks.append(mask)
            srcs.append(all_time_feats)

        # position encoding
        for x in out:
            pos.append(self.position_embedding(x).to(x.tensors.dtype))

        memory = self.transformer(srcs, masks, pos)
        b, _, c = memory.shape
        # (b,h1*w1+h2*w2...+hN*wN,c=hidden_dim)
        assert memory.shape[1] == get_in_size(self.feat_scales, downsample_times=self.downsample_times, mode='h*w')
        # (b,h,w,c)->(b,c,h,w)
        h0 = w0 = get_in_size([self.feat_scales[0]], downsample_times=self.downsample_times, mode='side')
        layer0 = memory[:, :h0 * w0, :].reshape(b, h0, w0, c).permute(0, 3, 1, 2)

        # pixel-wise Classifier
        pred = self.clfr_stg2(self.relu(self.clfr_stg1(layer0)))
        pred = F.interpolate(pred, size=(self.img_size, self.img_size), mode="bilinear")

        return pred
