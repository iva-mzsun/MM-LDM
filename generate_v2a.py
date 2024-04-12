import argparse
import glob
import os
import sys
import time
import numpy as np
from omegaconf import OmegaConf

import torch
from tqdm import tqdm

try:
    import lightning.pytorch as pl
    from lightning.pytorch import seed_everything
    from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.trainer import Trainer
    from lightning.pytorch.utilities import rank_zero_info, rank_zero_only
    LIGHTNING_PACK_NAME = "lightning.pytorch."
except:
    import pytorch_lightning as pl
    from pytorch_lightning import seed_everything
    from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.trainer import Trainer
    from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
    LIGHTNING_PACK_NAME = "pytorch_lightning."

from einops import repeat, rearrange
from src.utils.util import instantiate_from_config
from src.utils.save_mm import save_img, save_audio, save_multimodal
from main import get_parser, load_state_dict, nondefault_trainer_args

from ipdb import set_trace as st

def get_dataloader(data_cfg, batch_size, total_part, cur_part):
    import torch.utils.data as Data
    dataset = instantiate_from_config(data_cfg)
    assert len(dataset) == len(dataset.items)
    part_num = len(dataset) // total_part
    part_start = int((cur_part-1) * part_num)
    part_end = int(part_start + part_num)
    dataset.items = dataset.items[part_start:part_end]
    print(f"- Cur part {opt.cur_part}/{opt.total_part} "
          f"with items from {part_start} to {part_end}")
    dataloader = Data.DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        drop_last=False
    )
    return dataloader


if __name__ == "__main__":
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    opt, unknown = parser.parse_known_args()
    # init and save configs
    configs = [OmegaConf.load(cfg.strip()) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())

    # set save directories
    if opt.resume:
        rank_zero_info("Resuming from {}".format(opt.resume))
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            logdir = "/".join(paths[:-2])
            rank_zero_info("logdir: {}".format(logdir))
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        raise  NotImplementedError

    ckpt_name = os.path.basename(ckpt).split(".")[0]
    if opt.ucgs_audio_ae is not None:
        config.model.params.stable_ae_config.params.decode_sample_cfg.ucgs_audio = opt.ucgs_audio_ae
        config.model.params.stable_ae_config.params.decode_sample_cfg.ucgs_video = opt.ucgs_video_ae
        if opt.suffix is None:
            opt.suffix = f"aev{opt.ucgs_video_ae}_aea{opt.ucgs_audio_ae}"
        else:
            opt.suffix = f"{opt.suffix}-aev{opt.ucgs_video_ae}_aea{opt.ucgs_audio_ae}"
    if opt.suffix is None or opt.suffix=="":
        sampledir_video = os.path.join(logdir, f'samples-videos-{ckpt_name}-ucgs{opt.ucgs_video}')
        sampledir_audio = os.path.join(logdir, f'samples-audios-{ckpt_name}-ucgs{opt.ucgs_audio}')
        sampledir_mm = os.path.join(logdir, f'samples-mm-{ckpt_name}-ucvid{opt.ucgs_video}-ucaud{opt.ucgs_audio}')
    else:
        sampledir_video = os.path.join(logdir, f'samples-videos-{ckpt_name}-ucgs{opt.ucgs_video}-{opt.suffix}')
        sampledir_audio = os.path.join(logdir, f'samples-audios-{ckpt_name}-ucgs{opt.ucgs_audio}-{opt.suffix}')
        sampledir_mm = os.path.join(logdir, f'samples-mm-{ckpt_name}-ucvid{opt.ucgs_video}-ucaud{opt.ucgs_audio}-{opt.suffix}')
    os.makedirs(sampledir_mm, exist_ok=True)
    os.makedirs(sampledir_video, exist_ok=True)
    os.makedirs(sampledir_audio, exist_ok=True)
    seed_everything(opt.seed)

    # merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    trainer_config["devices"] = opt.ngpu or trainer_config["devices"]
    print(f"!!! WARNING: Number of gpu is {trainer_config['devices']} ")
    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    # model
    config.model["params"].update({"use_fp16": False})
    model = instantiate_from_config(config.model).cpu()
    # ckpt
    load_strict = trainer_config.get('ckpt_load_strict', True)
    ignore_keys = trainer_config.get('ignore_keys', [])
    state = load_state_dict(ckpt.strip(), ignore_keys=ignore_keys)
    model.load_state_dict(state, strict=load_strict)
    print(f"Load ckpt from {ckpt} with strict {load_strict}")
    model = model.cuda().eval()

    # data
    print(f"- Loading validation data...")
    bs = opt.batch_size or config.data.params.batch_size
    dataloader = get_dataloader(config.data.params.validation,
                                bs, opt.total_part, opt.cur_part)

    # start to generate
    verbose = opt.test_verbose
    save_mode = opt.save_mode # bybatch, byvideo, byframe
    ddim_step = opt.ddim_step
    ucgs_video = opt.ucgs_video
    ucgs_audio = opt.ucgs_audio
    video_length = opt.video_length
    total_sample_number = opt.total_sample_number
    audio_fps = config.data.params.validation.params.audio_fps
    print(f"- Saving generated video samples to {sampledir_video}")
    print(f"- Saving generated audio samples to {sampledir_audio}")
    print(f"- Saving generated multimedia samples to {sampledir_mm}")
    batch_idx = 0
    for batch in tqdm(iter(dataloader)):
        start = time.time()
        sample_log = model.log_samples_v2a(batch, N=bs, n_frames=video_length,
                                           verbose=verbose, sample=False, ddim_steps=ddim_step,
                                           ucgs_video=opt.ucgs_video, ucgs_audio=opt.ucgs_audio,
                                           ucgs_video_ae=opt.ucgs_video_ae, ucgs_audio_ae=opt.ucgs_audio_ae)
        print(f"sample time: {time.time() - start}")
        assert opt.ucgs_video != 0.0 and opt.ucgs_audio != 0.0
        video = sample_log[f"raw_video_frames"] # b c t h w
        # video = sample_log["video_rec"]
        video = ((video + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        video = rearrange(video, 'b c t h w -> b t h w c')
        all_videos = video.cpu().contiguous().numpy()
        all_audios, raw_audio_size = sample_log[f"audio_sample_ucgs{opt.ucgs_audio}"] # b c h w
        # all_audios, raw_audio_size = sample_log[f"audio_rec"] # b c h w

        item_names = batch['id']
        for b, name in enumerate(item_names):
            mm_name = f"b{batch_idx:04d}{b:02d}-v{name}-s{opt.seed}.mp4"
            video_name = f"b{batch_idx:04d}{b:02d}-v{name}-s{opt.seed}"
            audio_name = f"b{batch_idx:04d}{b:02d}-v{name}-s{opt.seed}.wav"

            mm_path = os.path.join(sampledir_mm, mm_name)
            video_path = os.path.join(sampledir_video, video_name)
            audio_path = os.path.join(sampledir_audio, audio_name)

            try:
                audio = model.audio_reconstructer.reconstruct(all_audios[b], raw_audio_size[b])
            except:
                audio = model.stable_ae.audio_reconstructer.reconstruct(all_audios[b], raw_audio_size[b])
            save_img(all_videos[b], video_path)
            save_audio(audio, audio_path, audio_fps=audio_fps)
            # model.audio_reconstructer.save_wave(audio[0], audio_path)
            save_multimodal(all_videos[b], audio, mm_path, video_fps=10, audio_fps=audio_fps)

        if len(os.listdir(sampledir_mm)) >= total_sample_number:
            final_number =len(os.listdir(sampledir_video))
            print(f"Having generated {final_number} video samples in {sampledir_video}!")
            exit()
        batch_idx += 1
