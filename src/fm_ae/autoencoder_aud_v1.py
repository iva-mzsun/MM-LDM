import einops
import torch
import torch.nn as nn
from einops import rearrange, repeat

from src.fm_ae.diffusion.ddim_v2 import DDIMSampler
from src.utils.util import instantiate_from_config
from src.modules.diffusionmodules.openaimodel import UNetModel
from src.modules.diffusionmodules.util import timestep_embedding
from src.fm_ae.diffusion.ddpm_v2 import LatentDiffusion
from ipdb import set_trace as st

class AEUnetModel(UNetModel):
    def forward(self, x_video, x_audio,
                timesteps_video, timesteps_audio,
                cond_feats_video, cond_feats_audio,
                context_video=None, context_audio=None, **kwargs):
        x = torch.cat([x_audio], dim=0)
        timesteps = torch.cat([timesteps_audio], dim=0)
        context = torch.cat([context_audio], dim=0)
        # obtain timestep embeddings
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
        # encode: insert vc feats
        hs = []
        h = x.type(self.dtype)
        for i, module in enumerate(self.input_blocks):
            h = module(h, emb, context)
            feats = torch.cat([cond_feats_audio[i]], dim=0)
            h = h + feats
            hs.append(h)
        # decode
        h = self.middle_block(h, emb, context)
        for i, module in enumerate(self.output_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)
        h = self.out(h)
        return dict(audio=h, video=x_video)

class AutoEncLDM(LatentDiffusion):
    def __init__(self,
                 # params for video
                 videoenc_config,
                 video_keyframe_key,
                 video_tarframe_key,
                 video_tarframe_ind_key,
                 # lambda_function,
                 # params for audio
                 audioenc_config,
                 audio_image_key,
                 audio_reconstructer_config,
                 kl_loss_weight=1e-6,
                 optimizer="adam", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = optimizer
        self.kl_loss_weight = kl_loss_weight
        # ========== params for video ==========
        # obtain video content
        self.video_keyframe_key = video_keyframe_key
        self.video_tarframe_key = video_tarframe_key
        self.video_tarframe_ind_key = video_tarframe_ind_key
        videoenc_config['params']['sdinput_block_ds'] = self.model.diffusion_model.input_block_ds
        videoenc_config['params']['sdinput_block_chans'] = self.model.diffusion_model.input_block_chans
        self.video_enc = instantiate_from_config(videoenc_config)
        # self.obtain_lambda = lambda t: obtain_lambda(t, lambda_function, self.use_fp16)
        # ========================================
        # ========== params for audio ==========
        # obtain audio content
        self.audio_image_key = audio_image_key
        audioenc_config['params']['sdinput_block_ds'] = self.model.diffusion_model.input_block_ds
        audioenc_config['params']['sdinput_block_chans'] = self.model.diffusion_model.input_block_chans
        self.audio_enc = instantiate_from_config(audioenc_config)
        self.audio_reconstructer = instantiate_from_config(audio_reconstructer_config)
        # ========================================
        # initialize modality embedding for video
        self.context_emb = nn.Parameter(torch.randn(77, 768) * 0.1)

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

    def get_input_video(self, batch, bs=None, repeat_c_by_T=True, T=None,
                        unconditional_version=False, *args, **kwargs):
        with torch.no_grad():
            # ========== get video inputs ==========
            videoenc_input = self.get_video_frames(batch, k=self.video_keyframe_key, keep_tdim=True, bs=bs)
            video_tarframes = self.get_video_frames(batch, k=self.video_tarframe_key, keep_tdim=False, bs=bs)
            video_tarframe_indexes = batch[self.video_tarframe_ind_key][:videoenc_input.shape[0]].to(self.device)
        c = repeat(self.context_emb, 'l c -> b l c', b=videoenc_input.shape[0])

        if self.training:
            mask = (torch.rand(c.shape[0]).to(self.device) < 0.1) * 1.0
            maskc = repeat(mask, 'b -> b d1 d2', d1=1, d2=1)
            maskv = repeat(mask, 'b -> b d1 d2 d3 d4', d1=1, d2=1, d3=1, d4=1)
            uc = torch.zeros_like(c)
            c = maskc * uc + (1 - maskc) * c
            uc_videoenc_input = torch.zeros_like(videoenc_input)
            videoenc_input = maskv * uc_videoenc_input + (1 - maskv) * videoenc_input

        if repeat_c_by_T:
            multi = T or video_tarframes.shape[0] // videoenc_input.shape[0]
            c = repeat(c, 'b l c -> b t l c', t=multi)
            c = repeat(c, 'b t l c -> (b t) l c')

        if unconditional_version: # for log video samples
            uc = torch.zeros_like(c)
            uc_videoenc_input = torch.zeros_like(videoenc_input)
            return None, dict(c_crossattn=[uc],
                              c_video=[uc_videoenc_input])
        else:
            return video_tarframes, dict(c_crossattn=[c],
                                         c_video=[videoenc_input],
                                         c_index=[video_tarframe_indexes])

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

    def get_input_audio(self, batch, bs=None,
                        unconditional_version=False, *args, **kwargs):
        with torch.no_grad():
            audio_image = self.get_audio_image(batch, k=self.audio_image_key, bs=bs)
        c = repeat(self.context_emb, 'l c -> b l c', b=audio_image.shape[0])

        if self.training:
            mask = (torch.rand(c.shape[0]).to(self.device) < 0.1) * 1.0
            maskc = repeat(mask, 'b -> b d1 d2', d1=1, d2=1)
            maska = repeat(mask, 'b -> b d1 d2 d3', d1=1, d2=1, d3=1)
            uc = torch.zeros_like(c)
            c = maskc * uc + (1 - maskc) * c
            uc_audioenc_input = torch.zeros_like(audio_image)
            audioenc_input = maska * uc_audioenc_input + (1 - maska) * audio_image
        else:
            audioenc_input = audio_image

        if unconditional_version:
            uc = torch.zeros_like(c)
            uc_audioenc_input = torch.zeros_like(audio_image)
            return None, dict(c_crossattn=[uc], c_audio=[uc_audioenc_input])
        else:
            return audio_image, dict(c_crossattn=[c], c_audio=[audioenc_input])

    def get_input(self, batch, bs=None, *args, **kwargs):
        ax, conda = self.get_input_audio(batch, bs, *args, **kwargs)
        vx, condv = self.get_input_video(batch, bs, *args, **kwargs)

        x = dict(audio=ax, video=vx)
        cond = dict(audio=conda, video=condv)
        return x, cond

    def forward(self, x, c, *args, **kwargs):
        t_audio = torch.randint(0, self.num_timesteps, (x['audio'].shape[0],), device=self.device).long()
        t_video = torch.randint(0, self.num_timesteps, (x['video'].shape[0],), device=self.device).long()
        t = dict(audio=t_audio, video=t_video)
        return self.p_losses(x, c, t, *args, **kwargs)

    def shared_step(self, batch, **kwargs):
        x, cond = self.get_input(batch)
        loss = self(x, cond, **kwargs)
        return loss

    def p_losses(self, x_start, cond, t, noise=None, returnx=False):
        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        # forward diffusion
        t_video = t['video']
        t_audio = t['audio']
        x_start_video = x_start['video']
        x_start_audio = x_start['audio']
        noise_video = torch.randn_like(x_start_video) # [bt c h w]
        noise_audio = torch.randn_like(x_start_audio) # [bt c h w]
        x_noisy_video = self.q_sample(x_start=x_start_video, t=t_video, noise=noise_video)
        x_noisy_audio = self.q_sample(x_start=x_start_audio, t=t_audio, noise=noise_audio)
        x_noisy = dict(video=x_noisy_video, audio=x_noisy_audio)

        # backward diffusion
        model_output, kl_loss = self.apply_model(x_noisy, t, cond)

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
        loss_dict.update({f'{prefix}/loss_simple_v': loss_simple_video.mean()})
        loss_dict.update({f'{prefix}/loss_simple_a': loss_simple_audio.mean()})

        logvar_t_video = self.logvar[t['video']].to(self.device)
        logvar_t_audio = self.logvar[t['audio']].to(self.device)
        loss_video = loss_simple_video / torch.exp(logvar_t_video) + logvar_t_video
        loss_audio = loss_simple_audio / torch.exp(logvar_t_audio) + logvar_t_audio
        loss = loss_video.mean() + loss_audio.mean()
        loss = self.l_simple_weight * loss

        loss_vlb_video = self.get_loss(model_output['video'], target['video'],
                                       mean=False).mean(dim=(1, 2, 3))
        loss_vlb_audio = self.get_loss(model_output['audio'], target['audio'],
                                       mean=False).mean(dim=(1, 2, 3))
        loss_vlb = loss_vlb_video.mean() + loss_vlb_audio.mean()
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        if kl_loss['video'] is not None:
            kl_loss_mean = kl_loss['video'].mean() + kl_loss['audio'].mean()
            loss += kl_loss_mean * self.kl_loss_weight
            loss_dict.update({f'{prefix}/kl_loss_v': kl_loss['video'].mean()})
            loss_dict.update({f'{prefix}/kl_loss_a': kl_loss['audio'].mean()})

        if returnx:
            if self.parameterization == "eps":
                x_recon_video = self.predict_start_from_noise(x_noisy['video'], t=t['video'],
                                                              noise=model_output['video'])
                x_recon_audio = self.predict_start_from_noise(x_noisy['audio'], t=t['audio'],
                                                              noise=model_output['audio'])
                x_recon = dict(video=x_recon_video, audio=x_recon_audio)
            else:
                raise NotImplementedError()
            return loss, loss_dict, x_start, x_recon
        return loss, loss_dict

    @torch.no_grad()
    def get_uc_vidae_input(self, batch):
        B, _, T = batch[self.videocontent_key].shape[:2]
        in_channel = self.videocontent_enc.in_channels
        image_size = self.videocontent_enc.image_size
        shape = (B, in_channel, T, image_size, image_size)
        uc_video_input = torch.zeros(shape).to(self.device)
        return uc_video_input

    def obtain_cond_feats(self, cond, task_id):
        if cond.get('cond_feats', None):
            return cond['cond_feats'], None

        if task_id == "video":
            cond_index = torch.cat(cond['c_index'], 0) # b, t
            cond_video = torch.cat(cond['c_video'], 0)  # b, c, t, h, w
            cond_feats, kl_loss = self.video_enc(x=cond_video, index=cond_index,
                                                is_training=self.training)
        elif task_id == "audio":
            cond_audio = torch.cat(cond['c_audio'], 0)
            cond_feats, kl_loss = self.audio_enc(x=cond_audio, is_training=self.training)
        else:
            raise NotImplementedError

        return cond_feats, kl_loss

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        t_video = t['video']
        t_audio = t['audio']
        cond_video = cond['video']
        cond_audio = cond['audio']
        x_noisy_video = x_noisy['video']
        x_noisy_audio = x_noisy['audio']

        cond_txt_video = torch.cat(cond_video['c_crossattn'], 1) # b, l, c
        cond_txt_audio = torch.cat(cond_audio['c_crossattn'], 1) # b, l, c
        cond_feats_video, kl_loss_video = self.obtain_cond_feats(cond_video, task_id="video")
        cond_feats_audio, kl_loss_audio = self.obtain_cond_feats(cond_audio, task_id="audio")
        kl_loss = dict(audio=kl_loss_audio, video=kl_loss_video)

        eps = diffusion_model(x_video=x_noisy_video,
                              x_audio=x_noisy_audio,
                              timesteps_video=t_video,
                              timesteps_audio=t_audio,
                              context_video=cond_txt_video,
                              context_audio=cond_txt_audio,
                              cond_feats_video=cond_feats_video,
                              cond_feats_audio=cond_feats_audio)

        return eps, kl_loss

    @torch.no_grad()
    def log_samples(self, batch, N=4, n_frames=4, ddim_steps=50, ddim_eta=0.0, sample=True,
                   verbose=False, ucgs_video=3.0, ucgs_audio=3.0, **kwargs):
        # initialize settings
        log = dict()
        use_ddim = ddim_steps is not None
        N = min(batch[self.audio_image_key].shape[0], N)

        # obtain audio cond feats
        _, audio_cond = self.get_input_audio(batch, N)
        _, uc_audio_cond = self.get_input_audio(batch, N, unconditional_version=True)
        cond_feats_a, _ = self.obtain_cond_feats(audio_cond, task_id="audio")
        ucond_feats_a, _ = self.obtain_cond_feats(uc_audio_cond, task_id="audio")
        audio_cond.update({"cond_feats": cond_feats_a})
        uc_audio_cond.update({"cond_feats": ucond_feats_a})
        # obtain video cond feats
        _, video_cond = self.get_input_video(batch, N, repeat_c_by_T=True, T=n_frames)
        _, uc_video_cond = self.get_input_video(batch, N, repeat_c_by_T=True, T=n_frames,
                                                unconditional_version=True)
        index = [torch.ones((N, 1)) * t / n_frames for t in range(n_frames)]
        index = torch.cat(index, dim=1).to(self.device)  # b t
        video_cond.update({"c_index": [index]})
        uc_video_cond.update({"c_index": [index]})
        cond_feats_v, _ = self.obtain_cond_feats(video_cond, task_id="video")
        ucond_feats_v, _ = self.obtain_cond_feats(uc_video_cond, task_id="video")
        video_cond.update({"cond_feats": cond_feats_v})
        uc_video_cond.update({"cond_feats": ucond_feats_v})
        torch.cuda.empty_cache()

        # start sampling
        raw_audioimg_size = batch['raw_audioimg_size'][:N]
        log['gt_audio_image'] = (batch[self.audio_image_key][:N],
                                 raw_audioimg_size)
        log['gt_video_frames'] = batch[self.video_keyframe_key][:N]
        if sample:
            cond = dict(audio=audio_cond, video=video_cond)
            x_samples, _ = self.sample_log(cond=cond, ddim=use_ddim,
                                           n_frame_per_video=n_frames, batch_size=N,
                                           ddim_steps=ddim_steps, eta=ddim_eta, verbose=verbose)
            torch.cuda.empty_cache()
            audio_image = self.decode_first_stage(x_samples['audio']) # B C H W
            video_frames = self.decode_first_stage(x_samples['video'])  # B C H W
            torch.cuda.empty_cache()
            log[f"audio_sample"] = (audio_image, raw_audioimg_size)
            log[f"video_sample"] = rearrange(video_frames, '(b t) c h w -> b c t h w',
                                             b=N, t=n_frames)
        if ucgs_audio > 0 or ucgs_video > 0:
            ucgs = dict(audio=ucgs_audio, video=ucgs_video)
            cond = dict(audio=audio_cond, video=video_cond)
            uc_cond = dict(audio=uc_audio_cond, video=uc_video_cond)
            x_samples, _ = self.sample_log(cond=cond, ddim=use_ddim,
                                             n_frame_per_video=n_frames, batch_size=N,
                                             ddim_steps=ddim_steps, eta=ddim_eta, verbose=verbose,
                                             unconditional_conditioning=uc_cond,
                                             unconditional_guidance_scale=ucgs)
            torch.cuda.empty_cache()
            audio_image = self.decode_first_stage(x_samples['audio'])  # B C H W
            video_frames = self.decode_first_stage(x_samples['video'])  # B C H W
            torch.cuda.empty_cache()
            log[f"audio_sample_ucgs{ucgs_audio}"] = (audio_image, raw_audioimg_size)
            log[f"video_sample_ucgs{ucgs_video}"] = rearrange(video_frames, '(b t) c h w -> b c t h w',
                                                              b=N, t=n_frames)
        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps,
                   n_frame_per_video, verbose=False, **kwargs):
        ddim_sampler = DDIMSampler(self)
        shape = (self.channels, self.image_size, self.image_size)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape,
                                                     cond, n_frame_per_video,
                                                     verbose=verbose, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = []
        params += [self.context_emb]
        params += list(self.video_enc.parameters())
        params += list(self.audio_enc.parameters())

        # add select params
        for n, p in self.model.diffusion_model.named_parameters():
            params.append(p)

        # count total params to optimize
        optimize_params = 0
        for param in params:
            optimize_params += param.numel()
        print(f"NOTE!!! {optimize_params/1e6:.3f}M params to optimize in TOTAL!!!")

        if self.optimizer == "hybridadam":
            raise NotImplementedError
        else:
            print("Load AdamW optimizer.")
            opt = torch.optim.AdamW(params, lr=lr)

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
