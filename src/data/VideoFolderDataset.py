import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from ipdb import set_trace as st
from random import shuffle

from src.data.utils import center_crop

class VideoFolderDataset(Dataset):
    def __init__(self, root_list, num_frames, flip_p=0.5, allow_point=True,
                 default_prompt="", fix_prompt=None, center_crop=True,
                 content_frame_idx=(0, 5, 10, 15), full_video_length=16,
                 sort_index=False, size=None, max_data_num=None):
        self.size = size
        self.flip_prob = flip_p
        self.root_list = root_list
        self.sort_index = sort_index
        self.num_frames = num_frames
        self.allow_point = allow_point
        self.center_crop = center_crop
        self.fix_prompt = fix_prompt
        self.default_prompt = default_prompt
        self.content_frame_idx = content_frame_idx
        self.full_video_length = full_video_length
        self.interpolation = Image.BICUBIC
        # load videos
        self.videos = []
        for root in root_list:
            ids_file = os.path.join(root, 'all_ids.json')
            if os.path.exists(ids_file):
                cids = json.load(open(ids_file, 'r'))
            else:
                cids = os.listdir(root)
            cids = [[root, cid] for cid in cids]
            self.videos += cids
        # load id2caption
        if self.fix_prompt is None:
            self.id2cap = dict({})
            for root in root_list:
                id2cap_file = os.path.join(root, 'all_id2cap.json')
                cid2cap = json.load(open(id2cap_file, 'r'))
                self.id2cap.update(cid2cap)
        # trunk videos
        if max_data_num is not None:
            shuffle(self.videos)
            self.videos = self.videos[:max_data_num]
        # create point
        self.vidpoint = dict({})
        for v in self.videos:
            self.vidpoint[v[1]] = 0

    def __len__(self):
        return len(self.videos)

    def load_img(self, img_path, if_flip):
        image = Image.open(img_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        if self.center_crop:
            image = center_crop(image)
        if self.size is not None:
            image = image.resize((self.size, self.size),
                                 resample=self.interpolation)
        if if_flip:
            image = F.hflip(image)
        image = np.array(image).astype(np.uint8)
        image = image.transpose((2, 0, 1))
        image = (image / 127.5 - 1.0).astype(np.float32)
        return image

    def get_frames(self, curv):
        frames = sorted(os.listdir(curv))
        frames = [os.path.join(curv, f)
                  for f in frames]
        return frames

    def update_point(self, curv, nframes):
        if self.allow_point is False:
            return
        cpoint = self.vidpoint[curv]
        self.vidpoint[curv] = cpoint + 1
        if cpoint + self.full_video_length >= nframes:
            self.vidpoint[curv] = 0

    def obtain_content_frame_idx(self, video_length):
        if self.full_video_length <= video_length:
            return self.content_frame_idx
        else:
            part = video_length // 3
            if part * 3 == video_length:
                return [0, part * 1 - 1, part * 2 - 1, part * 3 - 1]
            else:
                return [0, part * 1, part * 2, part * 3]

    def __skip_sample__(self, idx):
        if idx == len(self.videos) - 1:
            return self.__getitem__(0)
        else:
            return self.__getitem__(idx+1)

    def __random_sample__(self):
        idx = np.random.randint(0, len(self.videos))
        return self.__getitem__(idx)

    def __getitem__(self, idx):
        root, cid = self.videos[idx]
        curv = os.path.join(root, cid)
        vname = os.path.basename(curv)
        # obtain video name and prompt
        if self.fix_prompt:
            prompt = self.fix_prompt
        elif self.id2cap[cid] == "":
            prompt = self.default_prompt
        else:
            prompt = self.id2cap[cid]
        # determine if flip
        p = np.random.rand()
        if_flip = p < self.flip_prob
        # load video frames * point
        frames = self.get_frames(curv)
        if len(frames) < 4:
            return self.__random_sample__()
        point = self.vidpoint[cid]
        self.update_point(cid, len(frames))
        # Start to obtain cur batch
        try:
            # load full video frames
            content_frames = []
            for i in self.obtain_content_frame_idx(len(frames)):
                cframe = self.load_img(frames[point + i], if_flip)
                content_frames.append(cframe[:, np.newaxis, :, :])
            content_frames = np.concatenate(content_frames, axis=1)
            # random select target frame indexes
            video_length = min(self.full_video_length, len(frames))
            indexes = []
            for _ in range(self.num_frames):
                tarindex = np.random.randint(0, video_length)
                preindex = tarindex if tarindex == 0 else tarindex - 1
                indexes.append((tarindex, preindex))
            if self.sort_index:
                indexes = sorted(indexes, key=lambda x: x[0])
            tar_indexes = np.array([ind[0] for ind in indexes])
            pre_indexes = np.array([ind[1] for ind in indexes])

            # load target video frames
            tar_frames = []
            for ind in tar_indexes:
                tar_frame = self.load_img(frames[point + ind], if_flip)
                tar_frames.append(tar_frame[:, np.newaxis, :, :])
            tar_indexes = tar_indexes.astype(np.float) / video_length
            tar_frames = np.concatenate(tar_frames, axis=1)

            # load previous video frames
            pre_frames = []
            for ind in pre_indexes:
                pre_frame = self.load_img(frames[point + ind], if_flip)
                pre_frames.append(pre_frame[:, np.newaxis, :, :])
            pre_indexes = pre_indexes.astype(np.float) / video_length
            pre_frames = np.concatenate(pre_frames, axis=1)

        except:
            return self.__random_sample__()

        return dict({
            'txt': prompt, # str
            'video_name': vname, # str
            'tar_frames': tar_frames, # [c, t, h, w]
            'frame_index': tar_indexes, # [t]
            'pre_frames': pre_frames,
            'preframe_index': pre_indexes,
            'full_frame': content_frames, # [c, t, h, w]
        })

class VideoFolderDataset_Inference(Dataset):
    def __init__(self, caps_path, num_replication,
                 content_frame_idx=(0, 5, 10, 15), full_video_length=16,
                 size=None, **kwargs):
        self.size = size
        self.content_frame_idx = content_frame_idx
        self.full_video_length = full_video_length
        self.interpolation = Image.BICUBIC
        # load videos
        captions = []
        with open(caps_path, 'r') as f:
            for line in f.readlines():
                captions.append(line.strip())
        print(captions)
        videos = list(range(len(captions)))
        # replicate
        self.videos = []
        self.captions = []
        for i in range(num_replication):
            self.videos += videos
            self.captions += captions

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        cid = self.videos[idx]
        videoname = f"cap_{cid:02d}"
        tar_indexes = np.random.rand(2)
        tar_frames = np.random.rand(3, 2, self.size, self.size)
        content_frames = np.random.rand(3, len(self.content_frame_idx),
                                        self.size, self.size)
        # obtain video name and prompt
        prompt = self.captions[cid]

        return dict({
            'txt': prompt, # str
            'video_name': videoname, # str
            'tar_frames': tar_frames, # [c, t, h, w]
            'full_frame': content_frames, # [c, t, h, w]
            'frame_index': tar_indexes, # [t]
        })
