import numpy as np
import os
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

class DataLoader2d(Dataset):
    def __init__(self, ct_path, seg_path, train=True, random=False, black=True):
        self.train = train
        self.random = random
        self.black = black
        data_path = os.listdir(ct_path)
        self.ct_path = [os.path.join(ct_path, i) for i in data_path if 'volume' in i]
        self.ct_path.sort()
        self.ct_path = self.ct_path
        self.seg_path = [os.path.join(seg_path, i) for i in data_path if 'segmentation' in i]
        self.seg_path.sort()
        self.seg_path = self.seg_path

    def __getitem__(self, index):
        ct = sitk.GetArrayFromImage(sitk.ReadImage(self.ct_path[index]))
        seg = sitk.GetArrayFromImage(sitk.ReadImage(self.seg_path[index]))
        ct = ct.astype(np.float32)
        seg = (seg > 0.5).astype(np.uint8)
        ct = ct.clip(-200, 250)
        ct = ct - ct.min() / (ct.max() - ct.min())
        if self.train and self.random:
            if self.random > ct.shape[0]:
                crop = self.random - ct.shape[0]
                tmp = np.zeros((self.random, ct.shape[1], ct.shape[2]))
                idx = np.random.randint(0, crop)
                tmp[idx:idx+ct.shape[0]] = ct
                ct = tmp
                tmp[idx:idx+seg.shape[0]] = seg
                seg = tmp
                del(tmp)
            else:
                start_slice = np.random.randint(0, ct.shape[0] - self.random)
                end_slice = start_slice + self.random
                ct = ct[start_slice:end_slice]
                seg = seg[start_slice:end_slice]
        if self.train and self.black:
            t = np.where(np.sum(seg, (1, 2))!=0)[0]
            t = t.flatten()
            ct, seg = ct[t], seg[t]
        return torch.FloatTensor(ct), torch.LongTensor(seg)

    def __len__(self):
        return len(self.ct_path)
