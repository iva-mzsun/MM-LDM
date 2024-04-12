import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from src.modules.diffusionmodules.util import conv_nd
from src.fm_ae.utils.blocks import ResBlockwoEmb, \
    SpatioTemporalAttentionBlock, normalization, Downsample, \
    SpatialAttentionBlock, TemporalAttentionBlock

from ipdb import set_trace as st

class ImageDiscriminator(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        num_res_blocks,
        spatial_attention_resolutions,
        temporal_attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=3,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        legacy=True,
        temporal_embeddings=False
    ):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks

        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        # temporal embeddings
        # create learnable embeddings to distinguish pre/cur video frame
        self.temporal_embed = temporal_embeddings
        if temporal_embeddings:
            shape = (1, in_channels, 3, image_size, image_size)
            self.frame_embed = nn.Parameter(torch.randn(shape), requires_grad=True)

        # create input blocks
        self.input_blocks = nn.ModuleList([
             conv_nd(dims, in_channels, model_channels, 3, padding=1)
        ])
        input_block_chans = [model_channels]
        ds, ch = 1, model_channels
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlockwoEmb(
                        ch,
                        dropout,
                        out_channels=mult * model_channels,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in spatial_attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = num_head_channels
                    layers.append(
                        SpatioTemporalAttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                elif ds in temporal_attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = num_head_channels
                    layers.append(
                        TemporalAttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(nn.Sequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    nn.Sequential(
                        ResBlockwoEmb(
                            ch,
                            dropout,
                            out_channels=out_ch,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, out_channels=out_ch, dims=dims,
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        # obtain video content feature
        self.out = nn.Sequential(
            conv_nd(dims, ch, model_channels, 3, padding=1),
            normalization(model_channels),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(), # B, C
            nn.Linear(model_channels, 1)
        )

    def forward(self, x, **kwargs):
        '''
        x: [(b t), c, 3, h, w]
        '''
        h = x.type(self.dtype)
        if self.temporal_embed:
            h = h + self.frame_embed
        for module in self.input_blocks:
            h = module(h)
        logit = self.out(h).squeeze(1)
        return logit # B

class Discriminator(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=3,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        legacy=True,
        temporal_embeddings=False
    ):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        # temporal embeddings
        # create learnable embeddings to distinguish pre/cur video frame
        self.temporal_embed = temporal_embeddings
        if temporal_embeddings:
            shape = (1, in_channels, 2, image_size, image_size)
            self.frame_embed = nn.Parameter(torch.randn(shape), requires_grad=True)

        # create input blocks
        self.input_blocks = nn.ModuleList([
             conv_nd(dims, in_channels, model_channels, 3, padding=1)
        ])
        input_block_chans = [model_channels]
        ds, ch = 1, model_channels
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlockwoEmb(
                        ch,
                        dropout,
                        out_channels=mult * model_channels,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = num_head_channels

                    layers.append(
                        SpatioTemporalAttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(nn.Sequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    nn.Sequential(
                        ResBlockwoEmb(
                            ch,
                            dropout,
                            out_channels=out_ch,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, out_channels=out_ch, dims=dims,
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        # obtain video content feature
        self.out = nn.Sequential(
            conv_nd(dims, ch, model_channels, 3, padding=1),
            normalization(model_channels),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(), # B, C
            nn.Linear(model_channels, 1)
        )

    def forward(self, x, **kwargs):
        '''
        x: [(b t), c, 2, h, w]
        '''
        h = x.type(self.dtype)
        if self.temporal_embed:
            h = h + self.frame_embed
        for module in self.input_blocks:
            h = module(h)
        logit = self.out(h).squeeze(1)
        return logit # B
