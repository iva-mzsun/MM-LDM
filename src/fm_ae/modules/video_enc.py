import torch
import torch as th
import torch.nn as nn
from einops import rearrange, repeat

from src.modules.distributions.distributions import DiagonalGaussianDistribution
from src.modules.diffusionmodules.openaimodel import UNetModel, \
    TimestepEmbedSequential, ResBlock, Downsample, normalization, ResBlock2n
from src.modules.diffusionmodules.util import conv_nd, linear, \
    zero_module, timestep_embedding

from src.fm_ae.utils.blocks import ResBlockwoEmb, TemporalAttentionBlock, \
    SpatialAttentionBlock

from ipdb import set_trace as st

class VideoContentEnc(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        out_channels,
        model_channels,
        num_res_blocks,
        learnable_content,
        sdinput_block_ds,
        sdinput_block_chans,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        legacy=True,
        learnvar=True
    ):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.learnvar = learnvar
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
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
        self.learnable_content = learnable_content

        # create learnable embedding
        if self.learnable_content:
            shape = (in_channels, image_size, image_size)
            self.video_content = nn.Parameter(torch.randn(shape), requires_grad=True)
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
                        SpatialAttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
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
        if self.learnable_content:
            self.out_vc = nn.Sequential(
                conv_nd(dims, ch, self.out_channels, 3, padding=1),
                normalization(self.out_channels),
                nn.SiLU(),
                conv_nd(dims, self.out_channels, self.out_channels, 1)
            )
            if self.learnvar:
                self.out_vc_var = nn.Sequential(
                    conv_nd(dims, ch, self.out_channels, 3, padding=1),
                    normalization(self.out_channels),
                    nn.SiLU(),
                    zero_module(conv_nd(dims, self.out_channels, self.out_channels, 1))
                )
        else:
            self.out_vc = nn.Sequential(
                conv_nd(dims, ch, self.out_channels, 3, padding=1),
                normalization(self.out_channels),
                nn.SiLU(),
                nn.AdaptiveAvgPool3d((1, image_size // ds, image_size // ds)),
                conv_nd(dims, self.out_channels, self.out_channels, 1)
            )
            if self.learnvar:
                self.out_vc_var = nn.Sequential(
                    conv_nd(dims, ch, self.out_channels, 3, padding=1),
                    normalization(self.out_channels),
                    nn.SiLU(),
                    nn.AdaptiveAvgPool3d((1, image_size // ds, image_size // ds)),
                    zero_module(conv_nd(dims, self.out_channels, self.out_channels, 1))
                )

        # embed index embed
        index_embed_dim = model_channels * 4
        self.index_embed = nn.Sequential(
            linear(model_channels, index_embed_dim),
            nn.SiLU(),
            linear(index_embed_dim, index_embed_dim),
        )

        # obtain output blocks
        self.output_blocks = nn.ModuleList([])
        for i, ch in enumerate(sdinput_block_chans):
            curds, n = ds, 0
            tards = sdinput_block_ds[i]
            if tards == ds:
                layers = [ResBlock(out_channels, index_embed_dim, dropout, ch)]
            elif ds < tards:
                while curds < tards:
                    n += 1
                    curds *= 2
                layers = [ResBlock2n(out_channels, index_embed_dim,
                                     dropout, ch, down=True, sample_num=n)]
            else:
                while curds > tards:
                    n += 1
                    curds /= 2
                layers = [ResBlock2n(out_channels, index_embed_dim,
                                     dropout, ch, up=True, sample_num=n)]
            layers.append(self.make_zero_conv(ch, dims=2))
            self.output_blocks.append(TimestepEmbedSequential(*layers))

            self.sdinput_block_ds = sdinput_block_ds
            self.sdinput_block_chans = sdinput_block_chans

    def make_zero_conv(self, channels, dims=None):
        dims = dims or self.dims
        return nn.Sequential(zero_module(conv_nd(dims, channels, channels, 1, padding=0)))

    def encode(self, x):
        if self.learnable_content:
            B, C, T, H, W = x.shape
            vc = repeat(self.video_content, 'c h w -> b c t h w', b=B, t=1)
            x = torch.cat([vc, x], dim=2)
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h)

        if self.learnable_content:
            h_mean = self.out_vc(h)
            vc_mean = h_mean[:, :, 0, :, :]  # B, C, H, W
            if self.learnvar:
                h_std = self.out_vc_var(h)
                vc_std = h_std[:, :, 0, :, :]
        else:
            vc_mean = self.out_vc(h)  # B, C, 1, H, W
            vc_mean = vc_mean.squeeze(2)  # B, C, H, W
            if self.learnvar:
                vc_std = self.out_vc_var(h)
                vc_std = vc_std.squeeze(2)
        if self.learnvar:
            vc_dist = torch.cat([vc_mean, vc_std], dim=1)
            vc_posterior = DiagonalGaussianDistribution(vc_dist)
            return vc_posterior # [B, C * 2, H, W]
        return vc_mean

    def decode(self, vc, index):
        """
            vc: B, C, H, W
            index: B, T
        """
        # resize vc
        B, T = index.shape
        vc = repeat(vc, 'b c h w -> b c t h w', t=T)
        vc = rearrange(vc, 'b c t h w -> (b t) c h w')
        # obtain index embedding
        index = rearrange(index, 'b t -> (b t)').to(vc.device)
        index_emb = timestep_embedding(index, self.model_channels)
        index_emb = self.index_embed(index_emb)
        # obtain insert feat
        vc_feats = []
        for module in self.output_blocks:
            vc_feat = module(vc, index_emb)
            vc_feats.append(vc_feat)
        return vc_feats

    def forward(self, x, index, is_training=False, return_latent=False, **kwargs):
        '''
            x: [b, c, t, h, w]
            ind: [b, t]
        '''
        vc_posterior = self.encode(x)
        if is_training is True and self.learnvar:
            vc = vc_posterior.sample()
            kl_loss = vc_posterior.kl()
        else:
            if self.learnvar:
                vc = vc_posterior.mode()
            else:
                vc = vc_posterior
            kl_loss = None
        vc_feats = self.decode(vc, index)
        if return_latent:
            return vc, vc_feats, kl_loss
        return vc_feats, kl_loss
