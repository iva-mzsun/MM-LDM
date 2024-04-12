import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from src.modules.distributions.distributions import DiagonalGaussianDistribution
from src.modules.diffusionmodules.openaimodel import UNetModel, \
    TimestepEmbedSequential, ResBlock, Downsample, normalization, ResBlock2n
from src.modules.diffusionmodules.util import conv_nd, linear, \
    zero_module, timestep_embedding

from src.fm_ae.utils.blocks import ResBlockwoEmb, TemporalAttentionBlock, \
    SpatialAttentionBlock

from ipdb import set_trace as st

class ClassifyHeadandLoss(nn.Module):
    def __init__(
        self,
        num_class,
        in_channels,
        dims=2
    ):
        super().__init__()
        self.num_class = num_class
        self.in_channels = in_channels

        self.map_video = nn.Sequential(
            conv_nd(dims, in_channels, in_channels, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, in_channels, in_channels, 3, padding=1)
        )
        self.map_audio = nn.Sequential(
            conv_nd(dims, in_channels, in_channels, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, in_channels, in_channels, 3, padding=1)
        )

        self.out = nn.Sequential(
            conv_nd(1, in_channels, in_channels, 1),
            normalization(in_channels),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
            conv_nd(1, in_channels, num_class, 1)
        )

        self.loss_criterion = nn.CrossEntropyLoss()

    def forward(self, z_video, z_audio, class_ids):
        """
            z*: [B, C, H'', W'']
            class_ids: [B, ]
        """
        z_video = self.map_video(z_video)
        z_audio = self.map_audio(z_audio)
        z_video = torch.flatten(z_video, start_dim=2)
        z_audio = torch.flatten(z_audio, start_dim=2)
        z = torch.cat([z_audio, z_video], dim=2) # [B, C, L]
        logits = self.out(z).squeeze()
        targets = F.one_hot(class_ids, self.num_class)
        targets = targets.to(torch.float).to(z_video.device)

        loss = self.loss_criterion(logits, targets)
        return loss

class ContrastiveHeadandLoss(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels,
        dims=2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels

        self.map_video = nn.Sequential(
            conv_nd(dims, in_channels, in_channels, 3, padding=1),
            normalization(in_channels),
            nn.SiLU(),
            conv_nd(dims, in_channels, in_channels, 3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            conv_nd(dims, in_channels, model_channels, 1)
        )
        self.map_audio = nn.Sequential(
            conv_nd(dims, in_channels, in_channels, 3, padding=1),
            normalization(in_channels),
            nn.SiLU(),
            conv_nd(dims, in_channels, in_channels, 3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            conv_nd(dims, in_channels, model_channels, 1)
        )

        self.contrastive_temp_v2a = nn.Parameter(torch.Tensor([0.07]))
        self.contrastive_temp_a2v = nn.Parameter(torch.Tensor([0.07]))

    def forward(self, z_video, z_audio, rank, gather_func):
        """
            z*: [B, C, H'', W'']
        """
        feat_video = self.map_video(z_video).squeeze() # [B, C]
        feat_audio = self.map_audio(z_audio).squeeze() # [B, C]
        feat_video = F.normalize(feat_video, dim=-1)
        feat_audio = F.normalize(feat_audio, dim=-1)

        video_feats_all = gather_func(feat_video) # [N, B, C]
        audio_feats_all = gather_func(feat_audio) # [N, B, C]
        video_feats_all = rearrange(video_feats_all, 'n b c -> (n b) c')
        audio_feats_all = rearrange(audio_feats_all, 'n b c -> (n b) c')

        bs = z_video.shape[0]
        N = video_feats_all.shape[0]
        sim_v2a = torch.matmul(feat_video, audio_feats_all.T)
        sim_v2a = sim_v2a / self.contrastive_temp_v2a
        sim_a2v = torch.matmul(feat_audio, video_feats_all.T)
        sim_a2v = sim_a2v / self.contrastive_temp_a2v
        targets = torch.linspace(rank*bs, rank*bs+bs-1, bs, dtype=torch.int64)
        targets = targets.to(z_video.device)

        loss = (
            F.cross_entropy(sim_a2v, targets) +
            F.cross_entropy(sim_v2a, targets)
        ) / 2.
        return loss







