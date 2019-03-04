import numpy as np
import os
import SimpleITK as sitk  # too slow!
import nibabel as nib
import torch
from torch.utils.data import Dataset

class DataLoader2d(Dataset):
    def __init__(self, ct_path, seg_path, train=True, random=False, black=True, test=False, mask=None):
        self.train = train
        self.test = test
        self.random = random
        self.black = black
        ct_list = os.listdir(ct_path)
        seg_list = os.listdir(seg_path)
        self.ct_path = [os.path.join(ct_path, i) for i in ct_list if 'volume' in i]
        self.ct_path.sort()
        self.seg_path = [os.path.join(seg_path, i) for i in seg_list if 'segmentation' in i]
        self.seg_path.sort()
        if mask:
            mask_list = os.listdir(mask)
            self.mask = [os.path.join(mask, i) for i in mask_list if 'volume' in i]
            self.mask.sort()
        else: 
            self.mask = mask


    def __getitem__(self, index):
        #ct = nib.load(self.ct_path[index]).get_data().transpose(2,1,0)
        #seg = nib.load(self.seg_path[index]).get_data().transpose(2,1,0)
        ct = sitk.GetArrayFromImage(sitk.ReadImage(self.ct_path[index]))
        seg = sitk.GetArrayFromImage(sitk.ReadImage(self.seg_path[index]))
        ct = np.array(ct, dtype=np.float32)
        seg = np.array(seg, dtype=np.uint8)
        ct = ct.clip(-200, 250)
        if self.mask:
            ct = ct * (seg > 0.5).astype(int)
            #mask = sitk.GetArrayFromImage(sitk.ReadImage(self.mask[index]))
            a1, a2, a3 = np.where(seg!=0)
            z_min, z_max = a1.min(), a1.max()
            x_min, x_max = a2.min(), a3.max()
            y_min, y_max = a3.min(), a3.max()
            x_max += 32 - (x_max - x_min) % 32
            y_max += 32 - (y_max - y_min) % 32
            ct, seg = ct[z_min:z_max, x_min:x_max, y_min:y_max], seg[z_min:z_max, x_min:x_max, y_min:y_max]
        ct = (ct - ct.min()) / (ct.max() - ct.min())
        ct = np.broadcast_to(ct, (3,)+ct.shape)
        if self.test:
            ct = np.array(ct)
            ct[0, 1:] = ct[1, 0:-1]
            ct[2, 0:-1] = ct[1, 1:]
            ct = ct.transpose(1,0,2,3)
            return torch.FloatTensor(ct), torch.LongTensor(seg), self.ct_path[index]
        ct = np.array([ct[0, 0:-2], ct[1, 1:-1], ct[2, 2:]])
        ct = ct.transpose(1,0,2,3)
        seg = seg[1:-1]
        if self.train and self.random:
            if self.random > ct.shape[0]:
                crop = self.random - ct.shape[1]
                tmp = np.zeros((self.random, 3, ct.shape[2], ct.shape[3]))
                idx = np.random.randint(0, crop)
                tmp[idx:idx+ct.shape[0]] = ct
                ct = tmp
                tmp[idx:idx+seg.shape[0]] = seg
                seg = tmp
                del(tmp)
            else:
                start_slice = np.random.randint(0, ct.shape[1] - self.random)
                end_slice = start_slice + self.random
                ct = ct[start_slice:end_slice]
                seg = seg[start_slice:end_slice]
        if self.train and self.black:
            #t = np.where(np.sum((seg>0.5), (1, 2))!=0)[0]
            #t = t.flatten()
            #ct, seg = ct[t], seg[t]
            a1, a2, a3 = np.where(seg!=0)
            z_min, z_max = a1.min(), a1.max()
            x_min, x_max = a2.min(), a3.max()
            y_min, y_max = a3.min(), a3.max()
            x_max += 32 - (x_max - x_min) % 32
            y_max += 32 - (y_max - y_min) % 32
            ct, seg = ct[z_min:z_max, x_min:x_max, y_min:y_max], seg[z_min:z_max, x_min:x_max, y_min:y_max]
        return torch.FloatTensor(ct), torch.LongTensor(seg)

    def __len__(self):
        return len(self.ct_path)
