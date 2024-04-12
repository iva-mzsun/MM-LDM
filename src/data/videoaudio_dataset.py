import os
import json
import torch
import os.path as P
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from ipdb import set_trace as st
from random import shuffle

from src.data.utils import Flip_Controller, Video_Pointer, Img_Loader

class VideoAudioDataset(Dataset):
    def __init__(self, ids_file,
                 video_manager_cfg,
                 audio_manager_cfg,
                 max_data_num=None):

        self.interpolation = Image.BICUBIC
        self.cids = json.load(open(ids_file, 'r'))
        self.video_manager = Video_Manager(self.cids, **video_manager_cfg)
        self.audio_manager = Audio_Manager(**audio_manager_cfg)

        # trunk data num
        if max_data_num is not None:
            shuffle(self.cids)
            self.cids = self.cids[:max_data_num]

    def __len__(self):
        return len(self.cids)

    def __skip_sample__(self, idx):
        if idx == len(self.cids) - 1:
            return self.__getitem__(0)
        else:
            return self.__getitem__(idx+1)

    def __random_sample__(self):
        idx = np.random.randint(0, len(self.cids))
        return self.__getitem__(idx)

    def __getitem__(self, idx):
        cid = self.cids[idx]
        video_item = self.video_manager[cid]
        stime = video_item['stime']
        etime = video_item['etime']
        audio_item = self.audio_manager.obtain_img_from_wav(cid, stime, etime)

        return dict(id=cid,
                    duration=etime-stime,
                    audio_img=audio_item['audio_img'],
                    raw_audioimg_size=audio_item['raw_audioimg_size'],
                    key_frames=video_item['key_frames'],
                    pre_frames=video_item['pre_frames'],
                    tar_frames=video_item['tar_frames'],
                    tar_frame_indexes=video_item['tar_frame_indexes'])

class Video_Manager(object):
    def __init__(self, cids, video_root, frame_size, FPS,
                 if_centercrop=True, flip_p=0.5, full_video_length=16,
                 content_frame_idx=(0, 5, 10, 15), disable_pointer=False):
        self.FPS = FPS
        self.frame_size = frame_size
        self.video_root = video_root
        self.if_centercrop = if_centercrop
        self.flip_controller = Flip_Controller(flip_p)
        self.disable_pointer = disable_pointer
        self.video_pointer = Video_Pointer(cids, full_video_length)
        self.content_frame_idx = content_frame_idx
        self.full_video_length = full_video_length
        self.load_img = Img_Loader(frame_size, if_centercrop).load_img

    def obtain_content_frame_idx(self, current_video_length):
        if self.full_video_length <= current_video_length:
            return self.content_frame_idx
        else:
            part = current_video_length / 3
            return [0, int(part * 1), int(part * 2),
                    current_video_length - 1]

    def __getitem__(self, cvid):
        vidpath = os.path.join(self.video_root, cvid)
        frames = [P.join(vidpath, f) for f in sorted(os.listdir(vidpath))]
        if len(frames) < len(self.content_frame_idx):
            return None # Return None if fail to load current item
        point = 0 if self.disable_pointer else \
            self.video_pointer.one_step(cvid, len(frames))
        if_flip = self.flip_controller.flip_flag()
        # current video clip properties
        current_video_length = min(self.full_video_length, len(frames))
        current_video_time = current_video_length / self.FPS
        start_time = point / self.FPS
        end_time = start_time + current_video_time
        # load key video frames
        key_frames = []
        for i in self.obtain_content_frame_idx(len(frames)):
            cframe = self.load_img(frames[point + i], if_flip)
            key_frames.append(cframe[:, np.newaxis, :, :])
        key_frames = np.concatenate(key_frames, axis=1)
        # load paired target and previous video frames
        tar_indexes = np.sort(np.random.randint(0, current_video_length, 2))
        tar_frames, pre_frames = [], []
        for ind in tar_indexes:
            tar_frame = self.load_img(frames[point + ind], if_flip)
            pre_frame = np.zeros_like(tar_frame) if ind == 0 else \
                        self.load_img(frames[point + ind], if_flip)
            tar_frames.append(tar_frame[:, np.newaxis, :, :])
            pre_frames.append(pre_frame[:, np.newaxis, :, :])
        tar_indexes = tar_indexes.astype(np.float) / current_video_length
        tar_frames = np.concatenate(tar_frames, axis=1)
        pre_frames = np.concatenate(pre_frames, axis=1)

        return dict(tar_frames=tar_frames, tar_frame_indexes=tar_indexes,
                    pre_frames=pre_frames, key_frames=key_frames,
                    stime=start_time, etime=end_time)

from src.modules.mel_audio_reconstructed.hifigan.meldataset import mel_spectrogram, \
    MAX_WAV_VALUE, load_wav, spectral_de_normalize_torch
class Audio_Manager(object):
    def __init__(self, audio_root, img_size, raw_size=None,
                 n_mels=80, sample_rate=16000, n_fft=1024, hop_size=256, win_size=1024, fmax=8000, fmin=0):
        self.audio_root = audio_root
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.raw_size = raw_size # original size of audio image before resizing to img_size
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmax = fmax
        self.fmin = fmin
        self.load_img = Img_Loader(img_size, if_centercrop=False).load_img

    def obtain_img_from_wav(self, cid, stime, etime):
        # obtain audio
        audio_path = P.join(self.audio_root, f"{cid}.wav")
        audio, sr = load_wav(audio_path)
        audio = audio[int(sr * stime):int(sr * etime)]
        # print(0, sr, stime, etime, np.array(audio).shape)

        # obtain melspectrogram
        if len(audio) < self.n_fft:
            padding = self.n_fft - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')

        Time = etime - stime
        if len(audio) < sr * Time:
            padding = int(sr * Time) - len(audio)
            audio = np.concatenate((audio, np.zeros(padding)))
        # print(1, np.array(audio).shape)

        audio = audio / MAX_WAV_VALUE
        audio = torch.FloatTensor(audio).unsqueeze(0)
        melspectrogram = mel_spectrogram(audio, self.n_fft, self.n_mels, self.sample_rate,
                                         self.hop_size, self.win_size, self.fmin, self.fmax)
        melspectrogram = spectral_de_normalize_torch(melspectrogram)
        mel_array = (torch.log10(melspectrogram) * 20.0 - 20.0 + 100.0) / 100.0
        mel_normalized = mel_array.clip(0.0, 1.0)
        # print(2, mel_normalized.shape)

        if isinstance(mel_normalized, torch.Tensor):
            mel_normalized = mel_normalized.numpy()
        img_data = np.squeeze(mel_normalized)
        # convert to RGB img
        audio_img = (img_data * 255).astype(np.uint8)
        audio_img = Image.fromarray(audio_img)
        # print(3, audio_img.size)
        raw_audioimg_size = np.array(audio_img.size)
        audio_img = self.load_img(audio_img, if_flip=False)
        # print(4, audio_img.shape)
        # print(5, raw_audioimg_size)
        return dict(audio_img=audio_img, raw_audioimg_size=raw_audioimg_size)

