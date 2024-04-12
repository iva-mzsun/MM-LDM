import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange


class gan_loss(object):
    def __init__(self, loss_type):
        super(gan_loss, self).__init__()
        self.loss_type = loss_type

    def __call__(self, logits, label):
        # label=1: real sample
        # label=0: fake sample
        if self.loss_type == "hinge":
            weight = label.detach() * 2 - 1  # 1: real, -1: fake
            d_loss = F.relu(1. - logits * weight)
        else:
            raise NotImplementedError
        return d_loss.mean()


class disc_manager(object):
    def __init__(self, type):
        super(disc_manager, self).__init__()
        self.type = type

    def set_decode_func(self, first_stage_decode_function):
        self.decode_func = first_stage_decode_function

    def default(self, realx, fakex, ifdisc):
        realaud = self.decode_func(realx['audio']) # B C H W
        fakeaud = self.decode_func(fakex['audio']) # B C H W
        realvid = self.decode_func(realx['video']) # BT C H W
        fakevid = self.decode_func(fakex['video']) # BT C H W

        B = realaud.shape[0]
        dtype = realaud.dtype
        realaud = realaud.unsqueeze(2)
        fakeaud = fakeaud.unsqueeze(2)
        realvid = rearrange(realvid, '(b t) c h w -> b c t h w', b=B)
        fakevid = rearrange(fakevid, '(b t) c h w -> b c t h w', b=B)

        samples, labels = [], []
        if ifdisc == 0: # train generator
            # True samples
            samples.append(torch.cat([fakevid, fakeaud], dim=2))
            labels.append(torch.ones(B).to(dtype))
        elif ifdisc == 1: # train discriminator
            # True samples
            samples.append(torch.cat([realvid, realaud], dim=2))
            labels.append(torch.ones(B).to(dtype))
            # False samples
            samples.append(torch.cat([fakevid, fakeaud], dim=2))
            labels.append(torch.zeros(B).to(dtype))
            samples.append(torch.cat([realvid, fakeaud], dim=2))
            labels.append(torch.zeros(B).to(dtype))
            samples.append(torch.cat([fakevid, realaud], dim=2))
            labels.append(torch.zeros(B).to(dtype))
        # concate all samples
        samples = torch.cat(samples, dim=0)  # (b * 3 or 4) c 2 h w
        labels = torch.cat(labels, dim=0).to(samples.device)  # (b * 3 or 4)
        return samples, labels

    def only_video(self, realx, fakex, ifdisc):
        realvid = self.decode_func(realx['video']) # BT C H W
        fakevid = self.decode_func(fakex['video']) # BT C H W

        B = realx['audio'].shape[0]
        dtype = realvid.dtype
        realvid = rearrange(realvid, '(b t) c h w -> b c t h w', b=B)
        fakevid = rearrange(fakevid, '(b t) c h w -> b c t h w', b=B)
        realvid0 = realvid[:, :, :1, :, :]
        realvid1 = realvid[:, :, -1:, :, :]
        fakevid0 = fakevid[:, :, :1, :, :]
        fakevid1 = fakevid[:, :, -1:, :, :]

        samples, labels = [], []
        if ifdisc == 0: # train generator
            # True samples
            samples.append(torch.cat([fakevid0, fakevid1], dim=2))
            labels.append(torch.ones(B).to(dtype))
        elif ifdisc == 1: # train discriminator
            # True samples
            samples.append(torch.cat([realvid0, realvid1], dim=2))
            labels.append(torch.ones(B).to(dtype))
            # False samples
            samples.append(torch.cat([fakevid0, fakevid1], dim=2))
            labels.append(torch.zeros(B).to(dtype))
            samples.append(torch.cat([fakevid0, realvid1], dim=2))
            labels.append(torch.zeros(B).to(dtype))
            samples.append(torch.cat([realvid0, fakevid1], dim=2))
            labels.append(torch.zeros(B).to(dtype))
            samples.append(torch.cat([fakevid1, fakevid0], dim=2))
            labels.append(torch.zeros(B).to(dtype))

        # concate all samples
        samples = torch.cat(samples, dim=0)  # (b * 3 or 4) c 2 h w
        labels = torch.cat(labels, dim=0).to(samples.device)  # (b * 3 or 4)
        return samples, labels

    def audio_video(self, realx, fakex, ifdisc):
        realaud = self.decode_func(realx['audio']) # B C H W
        fakeaud = self.decode_func(fakex['audio']) # B C H W
        realvid = self.decode_func(realx['video']) # BT C H W
        fakevid = self.decode_func(fakex['video']) # BT C H W

        B = realaud.shape[0]
        dtype = realaud.dtype
        realaud = realaud.unsqueeze(2) # B C 1 H W
        fakeaud = fakeaud.unsqueeze(2) # B C 1 H W
        realvid = rearrange(realvid, '(b t) c h w -> b c t h w', b=B)
        fakevid = rearrange(fakevid, '(b t) c h w -> b c t h w', b=B)
        realvid0 = realvid[:, :, :1, :, :]
        realvid1 = realvid[:, :, -1:, :, :]
        fakevid0 = fakevid[:, :, :1, :, :]
        fakevid1 = fakevid[:, :, -1:, :, :]

        samples, labels = [], []
        if ifdisc == 0: # train generator
            # True samples
            samples.append(torch.cat([fakeaud, fakevid0, fakevid1], dim=2))
            labels.append(torch.ones(B).to(dtype))
        elif ifdisc == 1: # train discriminator
            # True samples
            samples.append(torch.cat([realaud, realvid0, realvid1], dim=2))
            labels.append(torch.ones(B).to(dtype))
            # False samples
            samples.append(torch.cat([fakeaud, fakevid0, fakevid1], dim=2))
            labels.append(torch.zeros(B).to(dtype))
            samples.append(torch.cat([fakeaud, fakevid0, realvid1], dim=2))
            labels.append(torch.zeros(B).to(dtype))
            samples.append(torch.cat([fakeaud, realvid0, fakevid1], dim=2))
            labels.append(torch.zeros(B).to(dtype))
            samples.append(torch.cat([fakeaud, fakevid1, fakevid0], dim=2))
            labels.append(torch.zeros(B).to(dtype))

        # concate all samples
        samples = torch.cat(samples, dim=0)  # (b n) c 3 h w
        labels = torch.cat(labels, dim=0).to(samples.device)  # (b n)
        return samples, labels


    def __call__(self, realx, fakex, ifdisc, *args, **kwargs):
        assert ifdisc in [0, 1]
        if self.type == "default":
            return self.default(realx, fakex, ifdisc)
        elif self.type == "only_video":
            return self.only_video(realx, fakex, ifdisc)
        elif self.type == "audio_video":
            return self.audio_video(realx, fakex, ifdisc)
        else:
            raise NotImplementedError
