import os
import json
import torch
import os.path as P
import numpy as np
import blobfile as bf
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from ipdb import set_trace as st

def center_crop(im): # im: PIL.Image
    width, height = im.size   # Get dimensions
    new_width = min(width, height)
    new_height = min(width, height)
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return im

def _list_video_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["avi", "gif", "mp4"]:

            results.append(full_path)
        elif bf.isdir(full_path):

            results.extend(_list_video_files_recursively(full_path))
    return results

class Img_Loader(object):
    def __init__(self,  size, if_centercrop):
        self.size = size
        self.if_centercrop = if_centercrop

    def load_img(self, image, if_flip):
        if isinstance(image, str):
            image = Image.open(image)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        if if_flip:
            image = F.hflip(image)
        if self.if_centercrop:
            image = center_crop(image)
        image = image.resize((self.size, self.size),
                             resample=Image.BICUBIC)
        image = np.array(image).astype(np.uint8)
        image = image.transpose((2, 0, 1))
        image = (image / 127.5 - 1.0).astype(np.float32)
        return image

def load_video_frame(self, image, if_flip):
    if isinstance(image, str):
        image = Image.open(image)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    if if_flip:
        image = F.hflip(image)

    image = image.resize((self.size, self.size),
                         resample=Image.BICUBIC)
    image = np.array(image).astype(np.uint8)
    image = image.transpose((2, 0, 1))
    image = (image / 127.5 - 1.0).astype(np.float32)
    return image

class Flip_Controller(object):
    def __init__(self, prob):
        super(Flip_Controller, self).__init__()
        self.prob = prob

    def flip_flag(self):
        p = np.random.rand()
        return p < self.prob

class Video_Pointer(object):
    def __init__(self, cids, video_length):
        # create point
        self.vidpoint = dict({})
        for cid in cids:
            self.vidpoint[cid] = 0
        self.video_length = video_length

    def one_step(self, cvid, nframes):
        cpoint = self.vidpoint[cvid]
        self.vidpoint[cvid] = cpoint + 1
        if cpoint + self.video_length >= nframes:
            self.vidpoint[cvid] = 0
        return cpoint

