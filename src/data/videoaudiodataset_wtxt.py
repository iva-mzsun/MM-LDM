import os
import json
import torch
import os.path as P
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from random import shuffle
from src.data.utils import Flip_Controller, Video_Pointer, Img_Loader

from ipdb import set_trace as st

class VideoAudioDataset_wTXT(Dataset):
    def __init__(self, ids_file,
                 video_root, video_fps,
                 audio_root, audio_fps,
                 video_manager_cfg,
                 audio_manager_cfg,
                 ids2txt_file=None,
                 fix_prompt=None,
                 max_data_num=None):
        self.nframe = video_manager_cfg['full_video_length']
        self.interpolation = Image.BICUBIC
        self.video_fps = video_fps
        self.audio_fps = audio_fps
        ids = json.load(open(ids_file, 'r'))
        items = [] # [ (video_id, stime, etime), ...]
        for cid in ids:
            cvideo = P.join(video_root, cid)
            frames = os.listdir(cvideo)
            if len(frames) <= self.nframe:
                items.append((cid, 0, len(frames) / video_fps))
            else:
                items.extend([(cid, i / video_fps, (i + self.nframe) / video_fps)
                              for i in range(len(frames) - self.nframe + 1)])
        self.items = items
        self.video_manager = Video_Manager(video_root, video_fps,
                                           **video_manager_cfg)
        self.audio_manager = Audio_Manager(audio_root, audio_fps,
                                           **audio_manager_cfg)
        self.fix_prompt = fix_prompt
        if fix_prompt is None:
            self.id2txt = json.load(open(ids2txt_file, 'r'))

        # trunk data num
        if max_data_num is not None:
            # shuffle(self.items)
            self.items = self.items[:512:64]

    def __len__(self):
        return len(self.items)

    def __skip_sample__(self, idx):
        if idx == len(self.items) - 1:
            return self.__getitem__(0)
        else:
            return self.__getitem__(idx+1)

    def __random_sample__(self):
        idx = np.random.randint(0, len(self.items))
        return self.__getitem__(idx)

    def __getitem__(self, idx):
        cid, stime, etime = self.items[idx]
        try:
            prompt = self.fix_prompt if self.fix_prompt is not None else self.id2txt[cid]
            video_item = self.video_manager.obtain_video_sample(cid, stime, etime)
            audio_item = self.audio_manager.obtain_img_from_npy(cid, stime, etime)
            # print(cid, prompt)
        except Exception as e:
            print(f"Fail to load video {cid} with exception {e}")
            return self.__skip_sample__(idx)
            # return self.__random_sample__()

        return dict(id=cid,
                    txt=prompt,
                    duration=etime-stime,
                    audio_img=audio_item['audio_img'],
                    raw_audioimg_size=audio_item['raw_audioimg_size'],
                    all_frames=video_item['all_frames'],
                    key_frames=video_item['key_frames'],
                    pre_frames=video_item['pre_frames'],
                    tar_frames=video_item['tar_frames'],
                    tar_frame_indexes=video_item['tar_frame_indexes'])

class Video_Manager(object):
    def __init__(self, video_root, video_fps, img_size, flip_p=0.5,
                 num_tar_frames=2, full_video_length=16,
                 content_frame_idx=(0, 5, 10, 15), return_complete_video=False):
        self.video_root = video_root
        self.video_fps = video_fps
        self.frame_size = img_size
        self.num_tar_frames = num_tar_frames
        self.flip_controller = Flip_Controller(flip_p)
        self.content_frame_idx = content_frame_idx
        self.full_video_length = full_video_length
        self.return_complete_video = return_complete_video

    def obtain_content_frame_idx(self, current_video_length):
        if self.full_video_length <= current_video_length:
            return self.content_frame_idx
        else:
            part = current_video_length / 3
            return [0, int(part * 1), int(part * 2),
                    current_video_length - 1]

    def load_img(self, img_path, if_flip):
        image = Image.open(img_path).convert("RGB")
        if if_flip:
            image = F.hflip(image) # [h w c]
        w, h = image.size
        old_size = (h, w)
        ratio = min(float(self.frame_size) / (old_size[0]),
                    float(self.frame_size) / (old_size[1]))
        new_size = tuple([int(i * ratio) for i in old_size])
        pad_h = self.frame_size - new_size[0]
        pad_w = self.frame_size - new_size[1]
        top, bottom = pad_h // 2, pad_h - (pad_h // 2)
        left, right = pad_w // 2, pad_w - (pad_w // 2)
        # print(111, image.size)
        transform = T.Compose([T.Resize(new_size, interpolation=InterpolationMode.BICUBIC),
                               T.Pad((left, top, right, bottom))])
        image = transform(image)
        image = np.array(image).astype(np.uint8)
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32) / 127.5 - 1.0
        # print(222, image.shape)
        return image

    def obtain_video_sample(self, cvid, stime, etime):
        vidpath = os.path.join(self.video_root, cvid)
        frames = [P.join(vidpath, f) for f in sorted(os.listdir(vidpath))]
        if_flip = self.flip_controller.flip_flag()
        # current video clip properties
        start_pos = int(stime * self.video_fps)
        end_pos = int(etime * self.video_fps)
        nframes = end_pos - start_pos

        if self.return_complete_video:
            all_frames = []
            for i in range(nframes):
                cframe = self.load_img(frames[start_pos + i], if_flip)
                all_frames.append(cframe[:, np.newaxis, :, :])
            all_frames = np.concatenate(all_frames, axis=1)
        else:
            all_frames = torch.randn(1)

        # load key video frames
        key_frames = []
        for i in self.obtain_content_frame_idx(nframes):
            cframe = self.load_img(frames[start_pos + i], if_flip)
            key_frames.append(cframe[:, np.newaxis, :, :])
        key_frames = np.concatenate(key_frames, axis=1)
        # load paired target and previous video frames
        if self.num_tar_frames < nframes:
            tar_indexes = np.sort(np.random.randint(0, nframes, self.num_tar_frames))
        else:
            tar_indexes = np.arange(0, nframes)
        tar_frames, pre_frames = [], []
        for ind in tar_indexes:
            tar_frame = self.load_img(frames[start_pos + ind], if_flip)
            pre_frame = np.zeros_like(tar_frame) if ind == 0 else \
                        self.load_img(frames[start_pos + ind], if_flip)
            tar_frames.append(tar_frame[:, np.newaxis, :, :])
            pre_frames.append(pre_frame[:, np.newaxis, :, :])
        tar_indexes = tar_indexes.astype(np.float) / nframes
        tar_frames = np.concatenate(tar_frames, axis=1)
        pre_frames = np.concatenate(pre_frames, axis=1)

        return dict(tar_frames=tar_frames, tar_frame_indexes=tar_indexes,
                    pre_frames=pre_frames, key_frames=key_frames, all_frames=all_frames)

from src.modules.mel_audio_reconstructed.hifigan.meldataset import mel_spectrogram, \
    MAX_WAV_VALUE, spectral_de_normalize_torch
class Audio_Manager(object):
    def __init__(self, audio_root, audio_fps, img_size, audio_length, sample_rate=22050,
                 n_mels=80, n_fft=1024, hop_size=256, win_size=1024, fmax=8000, fmin=0):
        self.audio_root = audio_root
        self.audio_fps = audio_fps
        self.audio_length = audio_length
        self.fmax = fmax
        self.fmin = fmin
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.sample_rate = sample_rate
        self.load_img = Img_Loader(img_size, if_centercrop=False).load_img

    def obtain_img_from_npy(self, cid, stime, etime):
        # obtain audio
        audio_path = P.join(self.audio_root, f"{cid}.npy")
        raw_audio = np.load(audio_path)[0]
        audio = raw_audio[int(self.audio_fps * stime):
                      int(self.audio_fps * etime)]
        # print(333, len(audio), len(raw_audio), self.audio_fps, stime, etime)
        # obtain melspectrogram
        if len(audio) < self.n_fft:
            padding = self.n_fft - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')

        if len(audio) < self.audio_length:
            padding = self.audio_length - len(audio)
            audio = np.concatenate((audio, np.zeros(padding)))

        audio = audio / MAX_WAV_VALUE
        audio = torch.FloatTensor(audio).unsqueeze(0)
        melspectrogram = mel_spectrogram(audio, self.n_fft, self.n_mels, self.sample_rate,
                                         self.hop_size, self.win_size, self.fmin, self.fmax)
        melspectrogram = spectral_de_normalize_torch(melspectrogram)
        mel_array = (torch.log10(melspectrogram) * 20.0 - 20.0 + 100.0) / 100.0
        mel_normalized = mel_array.clip(0.0, 1.0)

        if isinstance(mel_normalized, torch.Tensor):
            mel_normalized = mel_normalized.numpy()
        img_data = np.squeeze(mel_normalized)
        # convert to RGB img
        audio_img = (img_data * 255).astype(np.uint8)
        audio_img = Image.fromarray(audio_img)
        # print(444, audio_img.size)
        raw_audioimg_size = np.array(audio_img.size)
        audio_img = self.load_img(audio_img, if_flip=False)
        # print(555, audio_img.shape)
        # print(666, raw_audioimg_size)
        return dict(audio_img=audio_img, raw_audioimg_size=raw_audioimg_size)

    def obtain_mel_from_npy(self, cid, stime, etime):
        # obtain audio
        audio_path = P.join(self.audio_root, f"{cid}.npy")
        raw_audio = np.load(audio_path)[0]
        audio = raw_audio[int(self.audio_fps * stime):
                      int(self.audio_fps * etime)]
        # print(333, len(audio), len(raw_audio), self.audio_fps, stime, etime)
        # obtain melspectrogram
        if len(audio) < self.n_fft:
            padding = self.n_fft - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')

        if len(audio) < self.audio_length:
            padding = self.audio_length - len(audio)
            audio = np.concatenate((audio, np.zeros(padding)))

        audio = audio / MAX_WAV_VALUE
        audio = torch.FloatTensor(audio).unsqueeze(0)
        melspectrogram = mel_spectrogram(audio, self.n_fft, self.n_mels, self.sample_rate,
                                         self.hop_size, self.win_size, self.fmin, self.fmax)
        melspectrogram = spectral_de_normalize_torch(melspectrogram)
        mel_array = (torch.log10(melspectrogram) * 20.0 - 20.0 + 100.0) / 100.0
        mel_normalized = mel_array.clip(0.0, 1.0)
        return mel_normalized

