import SimpleITK as sitk
import os
import numpy as np


def score(seg_path, res_path):
    print seg_path, res_path
    seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
    res = sitk.GetArrayFromImage(sitk.ReadImage(res_path))
    print seg.shape, res.shape
    seg = seg.reshape(-1) > 0.5
    res = res.reshape(-1)
    return 2.0 * (seg*res).sum() / (seg.sum() + res.sum() + 1e-14)

if __name__ == '__main__':
    name = os.listdir('./dataset/val')
    res_list = ['./results/' + i  for i in name if 'volume' in i]
    seg_list = ['./dataset/val/' + i  for i in name if 'seg' in i]
    res_list.sort()
    seg_list.sort()
    c = []
    for i in range(len(res_list)):
        m =  score(seg_list[i], res_list[i])
        c.append(m)
        print 'dice score: %2.4f' % m
    print 'mean dice: ', np.array(c).mean()

