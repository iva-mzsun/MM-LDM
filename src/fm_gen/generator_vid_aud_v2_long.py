import einops
import os
import torch
import torch as th
import torch.nn as nn
from omegaconf import OmegaConf

from tqdm import tqdm
from einops import rearrange, repeat
from src.fm_gen.diffusion.ddim_v1 import DDIMSampler
from src.fm_gen.diffusion.ddpm import LatentDiffusion
from src.utils.util import instantiate_from_config, zero_module

from ipdb import set_trace as st

from src.fm_gen.diffusion.dit_v2 import *

class Previous_Cond(DiT):
    def __init__(self, *args, **kwargs):
        super(Previous_Cond, self).__init__(*args, **kwargs)
        del self.cond_embedder_audio
        del self.cond_embedder_video
        self.zero_layers = nn.ModuleList([
            self.make_zero_linear(self.hidden_size) for _ in range(self.depth)
        ])

    def make_zero_linear(self, channels):
        layer = nn.Linear(channels, channels, bias=False)
        return nn.Sequential(zero_module(layer))

    def forward(self, x, t, context):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        c: (N, L, C) tensor of class labels
        """
        x_video, x_audio = x['video'], x['audio']

        x_audio = self.x_embedder_audio(x_audio) # (N, T, D), where T = H * W / patch_size ** 2
        x_video = self.x_embedder_video(x_video)

        x_audio = x_audio + \
                  self.pos_embed_audio + \
                  self.mod_embed_audio
        x_video = x_video + \
                  self.pos_embed_video + \
                  self.mod_embed_video

        eos_audio = repeat(self.eos_audio, 'l c -> b l c', b=x_audio.shape[0])
        eos_video = repeat(self.eos_video, 'l c -> b l c', b=x_video.shape[0])
        x = torch.cat([eos_audio, x_audio, eos_video, x_video], dim=1)

        c = self.c_embedder(context) # (N, L, D), txt representation
        x = torch.cat([c, x], dim=1)

        t = self.t_embedder(t)  # (N, D)

        intermediate_xs = []
        for i in range(self.depth):
            x = self.blocks[i](x, t)  # (N, T, D)
            ix = self.zero_layers[i](x)
            intermediate_xs.append(ix)
        return intermediate_xs

class AR_DiT(DiT):
    def forward(self, x, t, context, cond_x, intermediate_x=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        c: (N, L, C) tensor of class labels
        """
        cond_audio = cond_x['audio']
        cond_video = cond_x['video']
        x_video, x_audio = x['video'], x['audio']

        cond_audio = self.cond_embedder_audio(cond_audio)
        cond_video = self.cond_embedder_video(cond_video)

        x_audio = self.x_embedder_audio(x_audio) # (N, T, D), where T = H * W / patch_size ** 2
        x_video = self.x_embedder_video(x_video)

        x_audio = x_audio + cond_audio + \
                  self.pos_embed_audio + \
                  self.mod_embed_audio
        x_video = x_video + cond_video + \
                  self.pos_embed_video + \
                  self.mod_embed_video

        eos_audio = repeat(self.eos_audio, 'l c -> b l c', b=x_audio.shape[0])
        eos_video = repeat(self.eos_video, 'l c -> b l c', b=x_video.shape[0])
        x = torch.cat([eos_audio, x_audio, eos_video, x_video], dim=1)

        c = self.c_embedder(context) # (N, L, D), txt representation
        x = torch.cat([c, x], dim=1)

        t = self.t_embedder(t)  # (N, D)
        for i, block in enumerate(self.blocks):
            x = block(x, t)  # (N, T, D)
            if intermediate_x is not None and \
                    i < len(intermediate_x):
                x = x + intermediate_x[i]
        x = x[:, c.shape[1]:, :]
        x_audio = x[:, 1:1+self.x_embedder_audio.num_patches, :]
        x_video = x[:, -self.x_embedder_video.num_patches:, :]
        x_audio = self.final_layer_audio(x_audio, t)  # (N, T, patch_size ** 2 * out_channels)
        x_video = self.final_layer_video(x_video, t)  # (N, T, patch_size ** 2 * out_channels)
        x_audio = self.unpatchify_audio(x_audio)  # (N, out_channels, H, W)
        x_video = self.unpatchify_video(x_video)  # (N, out_channels, H, W)
        return dict(audio=x_audio, video=x_video)

class Generator_LDM(LatentDiffusion):
    def __init__(self,
                 stable_ae_config,
                 prevideo_enc_config,
                 audio_image_key,
                 video_keyframe_key,
                 pre_audio_image_key,
                 pre_video_keyframe_key,
                 image_size, image_channels,
                 audio_latent_size, audio_latent_channels,
                 video_latent_size, video_latent_channels,
                 # audio_reconstructer_config,
                 optimizer="adam", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_size = image_size
        self.image_channels = image_channels
        self.audio_latent_size = audio_latent_size
        self.video_latent_size = video_latent_size
        self.audio_latent_channels = audio_latent_channels
        self.video_latent_channels = video_latent_channels

        self.optimizer = optimizer
        self.audio_image_key = audio_image_key
        self.video_keyframe_key = video_keyframe_key
        self.pre_audio_image_key = pre_audio_image_key
        self.pre_video_keyframe_key = pre_video_keyframe_key

        self.prevideo_enc = instantiate_from_config(prevideo_enc_config)
        # video auto encoder
        self.stable_ae = instantiate_from_config(stable_ae_config)
        for p in self.stable_ae.parameters():
            p.requires_grad = False
        # initialize learnable context
        self.context_emb = nn.Parameter(torch.randn(77, 768) * 0.1)
        self.audio_emb = nn.Linear(16, 768)
        self.video_emb = nn.Linear(16, 768)

    @torch.no_grad()
    def get_video_frames(self, batch, k, keep_tdim=True, bs=None):
        # obtain original video frames
        frames = batch[k].to(self.device) # [b c t h w]
        if bs is not None:
            frames = frames[:bs]
        if self.use_fp16:
            frames = frames.to(memory_format=torch.contiguous_format).half()
        else:
            frames = frames.to(memory_format=torch.contiguous_format).float()
        T = frames.shape[2]
        frames = einops.rearrange(frames, 'b c t h w -> (b t) c h w')
        frames = self.encode_first_stage(frames)
        frames = self.get_first_stage_encoding(frames).detach()
        if keep_tdim:
            frames = rearrange(frames, '(b t) c h w -> b c t h w', t=T)
        return frames

    @torch.no_grad()
    def get_audio_image(self, batch, k, bs=None):
        # obtain original video frames
        audio_img = batch[k].to(self.device)  # [b c h w]
        if bs is not None:
            audio_img = audio_img[:bs]
        if self.use_fp16:
            audio_img = audio_img.to(memory_format=torch.contiguous_format).half()
        else:
            audio_img = audio_img.to(memory_format=torch.contiguous_format).float()
        audio_img = self.encode_first_stage(audio_img)
        audio_img = self.get_first_stage_encoding(audio_img).detach()
        return audio_img

    def get_input(self, batch, bs=None, *args, **kwargs):
        with torch.no_grad():
            # obtain inputs for video/audio encoder
            audioenc_input = self.get_audio_image(batch, k=self.audio_image_key, bs=bs)
            videoenc_input = self.get_video_frames(batch, k=self.video_keyframe_key, keep_tdim=True, bs=bs)
            audio_latent, video_latent = self.stable_ae.encode(audioenc_input, videoenc_input)
            # obtain previous audio/video clips for condition
            pre_audioenc_input = self.get_audio_image(batch, k=self.pre_audio_image_key, bs=bs)
            pre_videoenc_input = self.get_video_frames(batch, k=self.pre_video_keyframe_key, keep_tdim=True, bs=bs)
            pre_audio_latent, pre_video_latent = self.stable_ae.encode(pre_audioenc_input, pre_videoenc_input)
        # conditions
        c = repeat(self.context_emb, 'l c -> b l c',
                   b=audio_latent.shape[0])
        cond_audio = torch.zeros_like(audio_latent) # serve for a2v or v2a tasks
        cond_video = torch.zeros_like(video_latent)

        if self.training:
            B = c.shape[0]
            modify_audio = audio_latent.detach()
            modify_video = video_latent.detach()
            uc = torch.zeros_like(c)

            maskc = torch.rand(B).to(self.device) < 0.1
            maska = (torch.rand(B).to(self.device) < 0.05) & ~maskc # only for conditional c
            maskv = (torch.rand(B).to(self.device) < 0.05) & ~maskc
            mask_axorv = maska ^ maskv
            maska = maska & mask_axorv # avoid both a and v
            maskv = maskv & mask_axorv
            maskc = maskc[:, None, None] * 1.0
            maska = maska[:, None, None, None] * 1.0
            maskv = maskv[:, None, None, None] * 1.0

            c = maskc * uc + (1 - maskc) * c
            cond_audio = maska * modify_audio + (1 - maska) * cond_audio
            cond_video = maskv * modify_video + (1 - maskv) * cond_video

            p = torch.rand(1)
            uc_pre_audio_latent = torch.zeros_like(pre_audio_latent)
            uc_pre_video_latent = torch.zeros_like(pre_video_latent)
            if p < 0.2: # audio continuation
                pre_video_latent = uc_pre_video_latent
            elif p < 0.4: # video continuation
                pre_audio_latent = uc_pre_audio_latent
            else:
                pass

        if bs is not None:
            c = c[:bs]
            audio_latent = audio_latent[:bs]
            video_latent = video_latent[:bs]

        return dict(audio=audio_latent, video=video_latent), \
               dict(c_crossattn=[c], cond_audio=[cond_audio], cond_video=[cond_video],
                    pre_audio=[pre_audio_latent], pre_video=[pre_video_latent])

    def shared_step(self, batch, is_training=False, **kwargs):
        x, c = self.get_input(batch)
        loss = self(x, c)
        return loss

    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x['audio'].shape[0],), device=self.device).long()
        return self.p_losses(x, c, t, *args, **kwargs)

    def p_losses(self, x_start, cond, t, noise=None):
        x_start_audio = x_start['audio']
        x_start_video = x_start['video']
        noise_video = torch.randn_like(x_start_video)  # [b c h w]
        noise_audio = torch.randn_like(x_start_audio)  # [b c h w]
        x_noisy_video = self.q_sample(x_start=x_start_video, t=t, noise=noise_video)
        x_noisy_audio = self.q_sample(x_start=x_start_audio, t=t, noise=noise_audio)
        x_noisy = dict(video=x_noisy_video, audio=x_noisy_audio)

        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "eps":
            target_video = noise_video
            target_audio = noise_audio
            target=dict(video=target_video, audio=target_audio)
        else:
            raise NotImplementedError()

        loss_simple_video = self.get_loss(model_output['video'], target['video'],
                                          mean=False).mean([1, 2, 3])
        loss_simple_audio = self.get_loss(model_output['audio'], target['audio'],
                                          mean=False).mean([1, 2, 3])
        loss_simple = loss_simple_audio.mean() + loss_simple_video.mean()
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})
        loss_dict.update({f'{prefix}/loss_simple_v': loss_simple_video.mean()})
        loss_dict.update({f'{prefix}/loss_simple_a': loss_simple_audio.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss_video = loss_simple_video / torch.exp(logvar_t) + logvar_t
        loss_audio = loss_simple_audio / torch.exp(logvar_t) + logvar_t
        loss = loss_video.mean() + loss_audio.mean()
        loss = self.l_simple_weight * loss

        loss_vlb_video = self.get_loss(model_output['video'], target['video'],
                                       mean=False).mean(dim=(1, 2, 3))
        loss_vlb_audio = self.get_loss(model_output['audio'], target['audio'],
                                       mean=False).mean(dim=(1, 2, 3))
        loss_vlb = loss_vlb_video.mean() + loss_vlb_audio.mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        diffusion_model = self.model.diffusion_model
        pre_audio_latent = cond['pre_audio'][0]
        pre_video_latent = cond['pre_video'][0]
        context_video = rearrange(pre_video_latent, 'b c h w -> b (h w) c')
        context_audio = rearrange(pre_audio_latent, 'b c h w -> b (h w) c')
        context = torch.cat(cond['c_crossattn'], 1)
        context_video = self.video_emb(context_video)
        context_audio = self.audio_emb(context_audio)
        context = torch.cat([context, context_video, context_audio], 1)
        cond_audio = cond['cond_audio'][0]
        cond_video = cond['cond_video'][0]
        cond_x = dict(audio=cond_audio, video=cond_video)

        pre_video_latent = torch.cat([pre_video_latent, x_noisy['video']], dim=1)
        pre_audio_latent = torch.cat([pre_audio_latent, x_noisy['audio']], dim=1)
        pre_x = dict(video=pre_video_latent, audio=pre_audio_latent)
        intermediate_x = self.prevideo_enc(x=pre_x, t=t, context=context)
        eps = diffusion_model(x=x_noisy, t=t, context=context,
                              cond_x=cond_x, intermediate_x=intermediate_x)

        return eps

    @torch.no_grad()
    def decode_vc_parrallel(self, latents, uc_latents, n_frames, verbose=False):
        torch.cuda.empty_cache()
        x_samples = self.stable_ae.decode(latents, uc_latents, n_frames)
        torch.cuda.empty_cache()
        audio_image = self.decode_first_stage(x_samples['audio'])  # B C H W
        video_frames = self.decode_first_stage(x_samples['video'])  # B C H W
        video_frames = rearrange(video_frames, '(b t) c h w -> b c t h w',
                                 t=n_frames)
        torch.cuda.empty_cache()
        return dict(audio=audio_image, video=video_frames)

    @torch.no_grad()
    def log_samples(self, batch, N=8, x_T=None, n_frames=4, sample=True, ar_times=2,
                    use_ddim=True, ddim_steps=50, ddim_eta=0.0,
                    audio_continuation=False, video_continuation=False,
                    verbose=False, ucgs_video=9.0, ucgs_audio=9.0, **kwargs):
        N = min(batch[self.video_keyframe_key].shape[0], N)
        ddim_kwargs = {"use_ddim": use_ddim, "ddim_steps": ddim_steps, "ddim_eta": ddim_eta}
        # obtain inputs & conditions
        assert use_ddim == (ddim_steps > 0)
        # obtain ground truth latents
        audioenc_input = self.get_audio_image(batch, k=self.audio_image_key, bs=N)
        videoenc_input = self.get_video_frames(batch, k=self.video_keyframe_key, keep_tdim=True, bs=N)
        audio_latent, video_latent = self.stable_ae.encode(audioenc_input, videoenc_input)
        # obtain previous audio/video clips for condition
        pre_audioenc_input = self.get_audio_image(batch, k=self.pre_audio_image_key, bs=N)
        pre_videoenc_input = self.get_video_frames(batch, k=self.pre_video_keyframe_key, keep_tdim=True, bs=N)
        pre_audio_latent, pre_video_latent = self.stable_ae.encode(pre_audioenc_input, pre_videoenc_input)
        uc_pre_audio_latent = torch.zeros_like(pre_audio_latent)
        uc_pre_video_latent = torch.zeros_like(pre_video_latent)
        # obtain unconditional latents
        uc_videoenc_input = torch.zeros_like(videoenc_input)
        uc_audioenc_input = torch.zeros_like(audioenc_input)
        uc_audio_latent, uc_video_latent = self.stable_ae.encode(uc_audioenc_input, uc_videoenc_input)
        # obtain conditions
        context = repeat(self.context_emb, 'l c -> b l c', b=N)
        uc_context = torch.zeros_like(context)
        cond_audio = torch.zeros_like(audio_latent)
        cond_video = torch.zeros_like(video_latent)
        # log
        log = dict()
        raw_audioimg_size = batch['raw_audioimg_size'][:N]
        tar_video_frames = batch[self.video_keyframe_key][:N].to(self.device)
        pre_video_frames = batch['pre_all_frames'][:N].to(self.device)
        raw_video_frames = torch.cat([pre_video_frames, tar_video_frames], dim=2)
        log['raw_video_frames'] = raw_video_frames # b, c, t, h, w

        tar_audio_image = batch[self.audio_image_key][:N].to(self.device)
        pre_audio_image = batch[self.pre_audio_image_key][:N].to(self.device)
        raw_audio_image = [pre_audio_image, tar_audio_image]
        log['raw_audio_image'] = (raw_audio_image, raw_audioimg_size)

        tar_latents = dict(audio=audio_latent, video=video_latent)
        uc_latents = dict(audio=uc_audio_latent, video=uc_video_latent)
        pre_latents = dict(audio=pre_audio_latent, video=pre_video_latent)
        tar_samples = self.decode_vc_parrallel(tar_latents, uc_latents, n_frames)
        pre_samples = self.decode_vc_parrallel(pre_latents, uc_latents, n_frames)
        log['rec_audio'] = ([pre_samples['audio'], tar_samples['audio']], raw_audioimg_size)
        log['rec_video'] = torch.cat([pre_samples['video'], tar_samples['video']], dim=2)

        if sample:
            sample_audio, sample_video = [], []
            # sample the first clip
            new_cond = {"c_crossattn": [context], "cond_video": [cond_video], "cond_audio": [cond_audio],
                        "pre_audio": [uc_pre_audio_latent], "pre_video": [uc_pre_video_latent]}
            latents, _ = self.sample_log(cond=new_cond, batch_size=N, x_T=x_T,
                                         verbose=verbose, **ddim_kwargs)
            samples = self.decode_vc_parrallel(latents, uc_latents, n_frames)
            sample_audio.append(samples['audio'])
            sample_video.append(samples['video'])
            # sample the following clips
            for i in range(ar_times):
                new_cond = {"c_crossattn": [context], "cond_video": [cond_video], "cond_audio": [cond_audio],
                            "pre_audio": [latents['audio']], "pre_video": [latents['video']]}
                latents, _ = self.sample_log(cond=new_cond, batch_size=N, x_T=x_T,
                                             verbose=verbose, **ddim_kwargs)
                samples = self.decode_vc_parrallel(latents, uc_latents, n_frames)
                sample_audio.append(samples['audio'])
                sample_video.append(samples['video'])
            log["sample_audio"] = (sample_audio, raw_audioimg_size)
            log["sample_video"] = torch.cat(sample_video, dim=2)

        if ucgs_video >= 0.0:
            sample_audio, sample_video = [], []
            ucgs = dict(video=ucgs_video, audio=ucgs_audio)
            if audio_continuation:
                print("Audio continuation...")
                sample_video.append(torch.zeros_like(pre_video_frames))
                sample_audio.append(pre_audio_image)
                latents = {'audio': pre_audio_latent, 'video': uc_pre_video_latent}
            elif video_continuation:
                print("Video continuation...")
                sample_video.append(pre_video_frames)
                sample_audio.append(torch.zeros_like(pre_audio_image))
                latents = {'audio': uc_pre_audio_latent, 'video': pre_video_latent}
            else:
                print("Non continuation...")
                # unconditional sample the first clip
                new_cond = {"c_crossattn": [context], "cond_video": [cond_video], "cond_audio": [cond_audio],
                            "pre_audio": [uc_pre_audio_latent], "pre_video": [uc_pre_video_latent]}
                uc_cond = {"c_crossattn": [uc_context], "cond_video": [cond_video], "cond_audio": [cond_audio],
                           "pre_audio": [uc_pre_audio_latent], "pre_video": [uc_pre_video_latent]}
                latents, _ = self.sample_log(cond=new_cond, batch_size=N, x_T=x_T,
                                             unconditional_conditioning=uc_cond,
                                             unconditional_guidance_scale=ucgs,
                                             verbose=verbose, **ddim_kwargs)
                samples = self.decode_vc_parrallel(latents, uc_latents, n_frames)
                sample_audio.append(samples['audio'])
                sample_video.append(samples['video'])
            # sample the following clips
            for i in range(ar_times):
                new_cond = {"c_crossattn": [context], "cond_video": [cond_video], "cond_audio": [cond_audio],
                            "pre_audio": [latents['audio']], "pre_video": [latents['video']]}
                uc_cond = {"c_crossattn": [uc_context], "cond_video": [cond_video], "cond_audio": [cond_audio],
                            "pre_audio": [uc_pre_audio_latent], "pre_video": [uc_pre_video_latent]}
                latents, _ = self.sample_log(cond=new_cond, batch_size=N, x_T=x_T,
                                             unconditional_conditioning=uc_cond,
                                             unconditional_guidance_scale=ucgs,
                                             verbose=verbose, **ddim_kwargs)
                samples = self.decode_vc_parrallel(latents, uc_latents, n_frames)
                sample_audio.append(samples['audio'])
                sample_video.append(samples['video'])
            log[f"audio_sample_ucgs{ucgs_audio}"] = (sample_audio, raw_audioimg_size)
            log[f"video_sample_ucgs{ucgs_video}"] = torch.cat(sample_video, dim=2)

        return log

    @torch.no_grad()
    def log_samples_a2v(self, batch, N=8, x_T=None, n_frames=4, sample=True,
                        use_ddim=True, ddim_steps=50, ddim_eta=0.0,
                        verbose=False, ucgs_video=9.0, ucgs_audio=9.0, **kwargs):
        N = min(batch[self.video_keyframe_key].shape[0], N)
        ddim_kwargs = {"use_ddim": use_ddim, "ddim_steps": ddim_steps, "ddim_eta": ddim_eta}
        # obtain inputs & conditions
        assert use_ddim == (ddim_steps > 0)
        # obtain ground truth latents
        audioenc_input = self.get_audio_image(batch, k=self.audio_image_key, bs=N)
        videoenc_input = self.get_video_frames(batch, k=self.video_keyframe_key, keep_tdim=True, bs=N)
        audio_latent, video_latent = self.stable_ae.encode(audioenc_input, videoenc_input)
        # obtain unconditional latents
        uc_videoenc_input = torch.zeros_like(videoenc_input)
        uc_audioenc_input = torch.zeros_like(audioenc_input)
        uc_audio_latent, uc_video_latent = self.stable_ae.encode(uc_audioenc_input, uc_videoenc_input)
        # obtain conditions
        context = repeat(self.context_emb, 'l c -> b l c', b=N)
        uc_context = torch.zeros_like(context)
        cond_audio = audio_latent.detach()
        cond_video = torch.zeros_like(video_latent)
        new_cond = {"c_crossattn": [context], "cond_video": [cond_video], "cond_audio": [cond_audio]}
        uc_cond = {"c_crossattn": [uc_context], "cond_video": [cond_video], "cond_audio": [cond_audio]}
        # log
        log = dict()
        raw_audioimg_size = batch['raw_audioimg_size'][:N]
        raw_video_frames = batch[self.video_keyframe_key][:N]
        log['raw_video_frames'] = raw_video_frames.to(self.device) # b, c, t, h, w
        raw_audio_image = batch[self.audio_image_key][:N]
        log['raw_audio_image'] = (raw_audio_image.to(self.device), raw_audioimg_size)
        latents = dict(audio=audio_latent, video=video_latent)
        uc_latents = dict(audio=uc_audio_latent, video=uc_video_latent)
        rec_samples = self.decode_vc_parrallel(latents, uc_latents, n_frames)
        log['rec_audio'] = (rec_samples['audio'], raw_audioimg_size)
        log['rec_video'] = rec_samples['video']

        # sampling
        if sample:
            latents, _ = self.sample_log(cond=new_cond, batch_size=N, x_T=x_T,
                                         verbose=verbose, **ddim_kwargs)
            samples = self.decode_vc_parrallel(latents, uc_latents, n_frames)
            log["sample_audio"] = (samples['audio'], raw_audioimg_size)
            log["sample_video"] = rearrange(samples['video'], '(b t) c h w -> b c t h w',
                                            b=N, t=n_frames)

        if ucgs_video >= 0.0:
            ucgs = dict(video=ucgs_video, audio=ucgs_audio)
            latents, _ = self.sample_log(cond=new_cond, batch_size=N, x_T=x_T,
                                         unconditional_conditioning=uc_cond,
                                         unconditional_guidance_scale=ucgs,
                                         verbose=verbose, **ddim_kwargs)
            samples = self.decode_vc_parrallel(latents, uc_latents, n_frames)
            log[f"audio_sample_ucgs{ucgs_audio}"] = (samples['audio'], raw_audioimg_size)
            log[f"video_sample_ucgs{ucgs_video}"] = rearrange(samples['video'],
                                                              '(b t) c h w -> b c t h w',
                                                              b=N, t=n_frames)

        return log

    @torch.no_grad()
    def log_samples_v2a(self, batch, N=8, x_T=None, n_frames=4, sample=True,
                        use_ddim=True, ddim_steps=50, ddim_eta=0.0,
                        verbose=False, ucgs_video=9.0, ucgs_audio=9.0, **kwargs):
        N = min(batch[self.video_keyframe_key].shape[0], N)
        ddim_kwargs = {"use_ddim": use_ddim, "ddim_steps": ddim_steps, "ddim_eta": ddim_eta}
        # obtain inputs & conditions
        assert use_ddim == (ddim_steps > 0)
        # obtain ground truth latents
        audioenc_input = self.get_audio_image(batch, k=self.audio_image_key, bs=N)
        videoenc_input = self.get_video_frames(batch, k=self.video_keyframe_key, keep_tdim=True, bs=N)
        audio_latent, video_latent = self.stable_ae.encode(audioenc_input, videoenc_input)
        # obtain unconditional latents
        uc_videoenc_input = torch.zeros_like(videoenc_input)
        uc_audioenc_input = torch.zeros_like(audioenc_input)
        uc_audio_latent, uc_video_latent = self.stable_ae.encode(uc_audioenc_input, uc_videoenc_input)
        # obtain conditions
        context = repeat(self.context_emb, 'l c -> b l c', b=N)
        uc_context = torch.zeros_like(context)
        cond_audio = torch.zeros_like(audio_latent)
        cond_video = video_latent.detach()
        new_cond = {"c_crossattn": [context], "cond_video": [cond_video], "cond_audio": [cond_audio]}
        uc_cond = {"c_crossattn": [uc_context], "cond_video": [cond_video], "cond_audio": [cond_audio]}
        # log
        log = dict()
        raw_audioimg_size = batch['raw_audioimg_size'][:N]
        raw_video_frames = batch[self.video_keyframe_key][:N]
        log['raw_video_frames'] = raw_video_frames.to(self.device) # b, c, t, h, w
        raw_audio_image = batch[self.audio_image_key][:N]
        log['raw_audio_image'] = (raw_audio_image.to(self.device), raw_audioimg_size)
        latents = dict(audio=audio_latent, video=video_latent)
        uc_latents = dict(audio=uc_audio_latent, video=uc_video_latent)
        rec_samples = self.decode_vc_parrallel(latents, uc_latents, n_frames)
        log['rec_audio'] = (rec_samples['audio'], raw_audioimg_size)
        log['rec_video'] = rec_samples['video']

        # sampling
        if sample:
            latents, _ = self.sample_log(cond=new_cond, batch_size=N, x_T=x_T,
                                         verbose=verbose, **ddim_kwargs)
            samples = self.decode_vc_parrallel(latents, uc_latents, n_frames)
            log["sample_audio"] = (samples['audio'], raw_audioimg_size)
            log["sample_video"] = rearrange(samples['video'], '(b t) c h w -> b c t h w',
                                            b=N, t=n_frames)

        if ucgs_video >= 0.0:
            ucgs = dict(video=ucgs_video, audio=ucgs_audio)
            latents, _ = self.sample_log(cond=new_cond, batch_size=N, x_T=x_T,
                                         unconditional_conditioning=uc_cond,
                                         unconditional_guidance_scale=ucgs,
                                         verbose=verbose, **ddim_kwargs)
            samples = self.decode_vc_parrallel(latents, uc_latents, n_frames)
            log[f"audio_sample_ucgs{ucgs_audio}"] = (samples['audio'], raw_audioimg_size)
            log[f"video_sample_ucgs{ucgs_video}"] = rearrange(samples['video'],
                                                              '(b t) c h w -> b c t h w',
                                                              b=N, t=n_frames)

        return log


    @torch.no_grad()
    def sample_log(self, cond, batch_size, use_ddim, ddim_steps, x_T, verbose=False, **kwargs):
        ddim_sampler = DDIMSampler(self)
        shape_audio = (batch_size, self.audio_latent_channels,
                       self.audio_latent_size, self.audio_latent_size)
        shape_video = (batch_size, self.video_latent_channels,
                       self.video_latent_size, self.video_latent_size)
        noise_audio = torch.randn(shape_audio, device=self.device)
        noise_video = torch.randn(shape_video, device=self.device)
        shape = dict(audio=shape_audio, video=shape_video)
        noise = dict(audio=noise_audio, video=noise_video)
        x_T = x_T if x_T is None else noise
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape,
                                                     cond, x_T=x_T, verbose=verbose, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.diffusion_model.parameters())
        params += list(self.prevideo_enc.parameters())
        params += list(self.audio_emb.parameters())
        params += list(self.video_emb.parameters())
        params += [self.context_emb]

        # == count total params to optimize ==
        optimize_params = 0
        for param in params:
            optimize_params += param.numel()
        print(f"NOTE!!! {optimize_params/1e6:.3f}M params to optimize in TOTAL!!!")

        # == load optimizer ==
        if self.optimizer.lower() == "adam":
            print("Load AdamW optimizer !!!")
            opt = torch.optim.AdamW(params, lr=lr)
        else:
            raise NotImplementedError

        # == load lr scheduler ==
        if self.use_scheduler:
            from torch.optim.lr_scheduler import LambdaLR
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()

    # ===== sample ddpm =====
    def sample_ddpm(self, cond, batch_size, verbose=False, **kwargs):
        shape = (batch_size, self.vc_channels, self.vc_size, self.vc_size)
        noise = torch.randn(shape, device=self.device)
        # start sampling
        vc_sample = noise
        # for i in tqdm(list(reversed(range(0, self.num_timesteps)))):
        for i in list(reversed(range(0, self.num_timesteps))):
            curt = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            vc_sample = self.p_sample(vc_sample, cond, curt, shape)
        return vc_sample