import torch
import torch.nn as nn
import torch.functional as F
from collections import OrderedDict
from src.fm_ae.diffusion.ddpm import DDPM
from src.fm_ae.diffusion.ddim_v6 import DDIMSampler
from src.utils.util import instantiate_from_config
from ipdb import set_trace as st
from einops import repeat, rearrange

class StableAutoencoder(DDPM):
    def __init__(self,
                 # params for video
                 videoenc_config,
                 video_projector,
                 # params for audio
                 audioenc_config,
                 audio_projector,
                 audio_reconstructer_config,
                 # params for heads
                 classify_head_cfg,
                 contrastive_head_cfg,
                 decode_sample_cfg, ckpt_path,
                 ignore_keys=[],
                 *args, **kwargs):
        super(StableAutoencoder, self).__init__(*args, **kwargs)
        self.num_class = self.model.diffusion_model.num_class
        self.classify_head = instantiate_from_config(classify_head_cfg)
        self.contrastive_head = instantiate_from_config(contrastive_head_cfg)
        # ========== params for video ==========
        # obtain video content
        videoenc_config['params']['sdinput_block_ds'] = self.model.diffusion_model.input_block_ds
        videoenc_config['params']['sdinput_block_chans'] = self.model.diffusion_model.input_block_chans
        self.video_enc = instantiate_from_config(videoenc_config)
        self.video_projector = instantiate_from_config(video_projector)
        # initialize modality embedding for video
        self.video_emb = nn.Parameter(torch.randn(77, 768) * 0.1)
        # self.obtain_lambda = lambda t: obtain_lambda(t, lambda_function, self.use_fp16)
        # ========================================
        # ========== params for audio ==========
        # obtain audio content
        audioenc_config['params']['sdinput_block_ds'] = self.model.diffusion_model.input_block_ds
        audioenc_config['params']['sdinput_block_chans'] = self.model.diffusion_model.input_block_chans
        self.audio_enc = instantiate_from_config(audioenc_config)
        self.audio_projector = instantiate_from_config(audio_projector)
        self.audio_reconstructer = instantiate_from_config(audio_reconstructer_config)
        # initialize modality embedding for audio
        self.audio_emb = nn.Parameter(torch.randn(77, 768) * 0.1)
        # ========================================
        self.shape_audio = None
        self.shape_video = None
        self.ckpt_path = ckpt_path
        if ckpt_path is not None:
            self.init_from_ckpt(path=ckpt_path, ignore_keys=ignore_keys)
        print(f"Restore StableAutoEncoder from {ckpt_path}")
        for p in self.parameters():
            p.require_grad = False
        self.decode_sample_cfg = decode_sample_cfg

    @torch.no_grad()
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        t_video = t['video']
        t_audio = t['audio']
        cond_video = cond['video']
        cond_audio = cond['audio']
        x_noisy_video = x_noisy['video']
        x_noisy_audio = x_noisy['audio']

        cond_txt_video = torch.cat(cond_video['c_crossattn'], 1) # b, l, c
        cond_txt_audio = torch.cat(cond_audio['c_crossattn'], 1) # b, l, c
        cond_feats_video = cond_video['cond_feats']
        cond_feats_audio = cond_audio['cond_feats']
        video_latent = cond_video['cond_latent']
        audio_latent = cond_audio['cond_latent']

        z_video = self.video_projector(video_latent.detach())  # [B, C, H'', W'']
        z_audio = self.audio_projector(audio_latent.detach())  # [B, C, H'', W'']
        cur_class_ids = self.classify_head.predict(z_video, z_audio)

        diffusion_model = self.model.diffusion_model
        eps = diffusion_model(class_ids=cur_class_ids,
                              x_video=x_noisy_video,
                              x_audio=x_noisy_audio,
                              z_video=z_video,
                              z_audio=z_audio,
                              video_latent=video_latent,
                              audio_latent=audio_latent,
                              timesteps_video=t_video,
                              timesteps_audio=t_audio,
                              context_video=cond_txt_video,
                              context_audio=cond_txt_audio,
                              cond_feats_video=cond_feats_video,
                              cond_feats_audio=cond_feats_audio)
        return eps, None

    @torch.no_grad()
    def encode(self, x_audio, x_video):
        '''
            x_audio: b, c, h, w
            x_video: b, c, t, h, w
        '''
        if self.shape_audio is None:
            self.shape_audio = list(x_audio.shape[1:])
        if self.shape_video is None:
            self.shape_video = list(x_video.shape[1:])

        z_audio = self.audio_enc.encode(x_audio)
        z_video = self.video_enc.encode(x_video)

        z_audio = z_audio.mode()
        z_video = z_video.mode()

        return z_audio, z_video

    @torch.no_grad()
    def decode(self, latents, uc_latents, n_frames=4, verbose=False):
        '''
            vc: b, c, h, w
            index: b, t
            context: b, l, c
        '''
        # obtain audio cond feats
        audio_latent = latents['audio']
        uc_audio_latent = uc_latents['audio']
        cond_feats_a = self.audio_enc.decode(audio_latent)
        ucond_feats_a = self.audio_enc.decode(uc_audio_latent)
        c_audio = repeat(self.audio_emb, 'l c -> b l c',
                         b=audio_latent.shape[0])
        uc_audio = torch.zeros_like(c_audio)
        cond_audio = dict(c_crossattn=[c_audio],
                          cond_feats=cond_feats_a,
                          cond_latent=audio_latent)
        ucond_audio = dict(c_crossattn=[uc_audio],
                           cond_feats=ucond_feats_a,
                           cond_latent=uc_audio_latent)

        # indexes of target video frames
        N = audio_latent.shape[0]
        index = [torch.ones((N, 1)) * t / n_frames
                 for t in range(n_frames)]
        index = torch.cat(index, dim=1).to(self.device)

        # obtain video cond feats
        video_latent = latents['video']
        uc_video_latent = uc_latents['video']
        cond_feats_v = self.video_enc.decode(video_latent, index)
        ucond_feats_v = self.video_enc.decode(uc_video_latent, index)
        c_video = repeat(self.video_emb, 'l c -> b l c',
                         b=video_latent.shape[0] * n_frames)
        uc_video = torch.zeros_like(c_video)
        cond_video = dict(c_crossattn=[c_video],
                          cond_feats=cond_feats_v,
                          cond_latent=video_latent)
        ucond_video = dict(c_crossattn=[uc_video],
                           cond_feats=ucond_feats_v,
                           cond_latent=uc_video_latent)

        # sampling
        ddim_sampler = DDIMSampler(self)
        use_ddim = self.decode_sample_cfg.use_ddim
        ddim_eta = self.decode_sample_cfg.ddim_eta
        ddim_steps = self.decode_sample_cfg.ddim_steps
        ucgs_audio = self.decode_sample_cfg.ucgs_audio
        ucgs_video = self.decode_sample_cfg.ucgs_video

        z_video = self.video_projector(video_latent.detach())  # [B, C, H'', W'']
        z_audio = self.audio_projector(audio_latent.detach())  # [B, C, H'', W'']
        class_ids = self.classify_head.predict(z_video, z_audio)

        ucgs = dict(audio=ucgs_audio, video=ucgs_video)
        cond = dict(audio=cond_audio, video=cond_video, c_class=class_ids)
        uc_cond = dict(audio=ucond_audio, video=ucond_video, c_class=class_ids)

        shape = (self.channels, self.image_size, self.image_size)
        samples, _ = ddim_sampler.sample(ddim_steps, N, shape,
                                         cond, n_frames, verbose=verbose,
                                         ddim=use_ddim, ddim_steps=ddim_steps, ddim_eta=ddim_eta,
                                         unconditional_conditioning=uc_cond,
                                         unconditional_guidance_scale=ucgs)

        return samples





