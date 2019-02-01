import torch
from torch import nn
from torch import optim
from importlib import import_module
from data import DataLoader2d as DatasetLoader 
from torch.utils.data import DataLoader
from torch.autograd import Variable
import SimpleITK as sitk
import numpy as np
import argparse
import time
import os


parser = argparse.ArgumentParser(description='U-Net 2d')
parser.add_argument('--resume', '-m', metavar='RESUME', default='',
                     help='model parameters to load')
parser.add_argument('--save_dir', default='', type=str, metavar='PATH',
                     help='path to save checkpoint files')
parser.add_argument('--test', default=0, type=int, metavar='TEST',
                     help='1 do test evaluation, 0 not')

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, out, seg):
        b, w, h = seg.shape
        seg = seg.unsqueeze(1)
        seg_one_hot = Variable(torch.FloatTensor(b,2, w, h)).zero_().cuda()
        seg = seg_one_hot.scatter_(1, seg, 1)
        loss = Variable(torch.FloatTensor(b)).zero_().cuda()
        for i in range(2):
            loss += (1 - 2.*((out[:,i]*seg[:,i]).sum(1).sum(1)) / ((out[:,i]*out[:,i]).sum(1).sum(1)+(seg[:,i]*seg[:,i]).sum(1).sum(1)+1e-15))
        loss = loss.mean()
        del seg_one_hot, seg
        return loss
    
def main():
    global args
    args = parser.parse_args()
    model = 'models.2d_unet'
    net = import_module(model).get_model()
    loss = DiceLoss()
    #loss = torch.nn.CrossEntropyLoss()
    #loss = SoftmaxLoss()
    net = net.cuda()
    loss = loss.cuda()
    net = torch.nn.DataParallel(net)
    if args.resume:
        checkpoint = torch.load(args.resume)
        net.module.load_state_dict(checkpoint['state_dict'])
    train_dataset = DatasetLoader('dataset/train', 
                               'dataset/train') #, random=64)
    val_dataset = DatasetLoader('dataset/val', 
                               'dataset/val', test=True)
    #val_dataset = DatasetLoader('/home/kxw/H-DenseUNet-master/data/myTestData', 
    #                                '/home/kxw/H-DenseUNet-master/livermask', train=False)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size = 1,
        shuffle = True,
        num_workers = 4,
        pin_memory=True)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = 1,
        pin_memory=True)

    if args.test == 1:
        test(val_loader, net, loss)
        return
    optimizer = optim.Adam(net.parameters(),
                        lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)

    def lr_restart(T0, Tcur, base_lr = 1e-3):
        lr_max = base_lr
        lr_min = base_lr * 1e-3
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1+np.cos(Tcur/float(T0) * np.pi))
        return lr

    T0 = 5
    Tcur = 0
    base_lr = 1e-3

    for epoch in range(1, 1000+1):
        print ("epoch", epoch)
        lr = lr_restart(T0, Tcur, base_lr)
        train(train_loader, net, loss, epoch, optimizer, lr, batch_size=24)
        #validate(val_loader, net, loss)

        Tcur = Tcur + 1
        if Tcur > T0:
            Tcur = 0
            T0 = T0 + 10
            base_lr = base_lr * 0.5

def train(train_loader, net, loss, epoch, optimizer, lr, batch_size):
    st = time.time()
    net.train()
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    losses = np.zeros(1)
    for i, (ct, seg) in enumerate(train_loader):
        #import pdb
        #pdb.set_trace()
        ct = ct.view(-1, 3, 512, 512)
        seg = seg.view(-1, 512, 512)
        seg = (seg > 0.5).long()
        for j in xrange(ct.shape[0]//batch_size):
            c = Variable(ct[j*batch_size:(j+1)*batch_size]).view(-1, 3, 512, 512).cuda()
            s = Variable(seg[j*batch_size:(j+1)*batch_size]).view(-1, 512, 512).cuda()
            out = net(c)
            loss_out = loss(out, s)
            optimizer.zero_grad()
            loss_out.backward()
            optimizer.step()
            losses += loss_out.data.cpu().numpy()
            del c, s, loss_out, out
    if epoch % 10 == 0:
        state_dict = net.module.state_dict()
        for key in state_dict:
            state_dict[key] = state_dict[key].cpu()
        torch.save({
            'epoch': epoch,
            'save_dir': './ckpt',
            'state_dict': state_dict},
            os.path.join('./ckpt', 'train_2d_%04d'%epoch+'.ckpt'))

    et = time.time()
    print('train loss %2.4f, time %2.4f' % (losses/101, et - st))

def validate(val_loader, net, loss):
    st = time.time()
    net.eval()
    losses = np.zeros(1)
    for i, (ct, seg) in enumerate(val_loader):
        ct = Variable(ct).cuda().view(-1, 512, 512)
        seg = Variable(seg).cuda().view(-1, 512, 512)
        #import pdb
        #pdb.set_trace()
        for j in xrange(ct.shape[0]):
            c = ct[j:(j+1)].view(-1, 1, 512, 512)
            s = seg[j:(j+1)].view(-1, 512, 512)
            out = net(c)
            loss_out = loss(out, s)
            losses += loss_out.data.cpu().numpy()
            del c, s, loss_out
        del ct, seg

    et = time.time()
    print('val loss %2.4f, time %2.4f' % (losses/70, et - st))

def test(val_loader, net, loss=None):
    st = time.time()
    net.eval()
    losses = np.zeros(1)
    softmax = nn.Softmax(dim=1)
    for i, (ct, seg, name) in enumerate(val_loader):
        out_results = []
        c1, c2 = 0, 0
        ct = Variable(ct).cuda().view(-1, 3, 512, 512)
        seg = Variable(seg).cuda().view(-1, 512, 512)
        for j in xrange(ct.shape[0]):
            c = ct[j:(j+1)].view(-1, 3, 512, 512)
            s = seg[j:(j+1)].view(-1, 512, 512)
            out = net(c)
            loss_out = loss(out, s)
            losses += loss_out.data.cpu().numpy()
            v, p = torch.max(softmax(out), 1)
            out_results.append(p.data.cpu().numpy()[0])
            p = p.flatten()
            s = s.flatten()
            c1 += 2.0 * (p*s).sum().data.cpu().numpy()
            c2 += (p.sum().data.cpu().numpy() + s.sum().data.cpu().numpy())
            del c, s, loss_out, v, p
        results = np.array(out_results)
        out = sitk.GetImageFromArray(results)
        sitk.WriteImage(out, './results/'+name[0].split('/')[-1])

        print name[0].split('/')[-1]
        print results.shape
        del ct, seg, out, results
        c = c1 / (c2 + 1e-14)
        print 'dice score', c


    et = time.time()
    print('test loss %2.4f, time %2.4f' % (losses/28, et - st))

if __name__ == '__main__':
    main()
