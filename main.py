import torch
from torch import nn
from torch import optim
from importlib import import_module
from data import DataLoader2d as DatasetLoader 
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import time
import os

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, out, seg):
        out = self.softmax(out)
        out_v, out_p = torch.max(out, dim=1)
        out_p = out_p.float()
        loss = torch.Variable(0)
        for i in xrange(out.shape[1]):
            loss += 1 - 2.*((out[:,i]*(seg==i)).sum(1).sum(1).sum(1)+1) / (out[:,i].sum(1).sum(1).sum(1)+(seg==i).sum(1).sum(1).sum(1)+1)
        return loss
class SoftmaxLoss(nn.Module):
    def __init__(self):
        super(SoftmaxLoss, self).__init__()
        self.loss = nn.NLLLoss()

    def forward(self, out, seg):
        import pdb
        pdb.set_trace()
        out = torch.nn.functional.log_softmax(out)
        loss = self.loss(out, seg[:,0])
        return loss
    

def main():
    import pdb
    # pdb.set_trace()
    model = 'models.2d_unet'
    net = import_module(model).get_model()
    # loss = DiceLoss()
    loss = torch.nn.CrossEntropyLoss()
    #loss = SoftmaxLoss()
    net = net.cuda()
    loss = loss.cuda()
    net = torch.nn.DataParallel(net)
    train_dataset = DatasetLoader('/home/kxw/H-DenseUNet-master/data/myTrainingData', 
                                    '/raid_1/data/liver/seg') #, random=64)
    val_dataset = DatasetLoader('/home/kxw/H-DenseUNet-master/data/myTestData', 
                                    '/home/kxw/H-DenseUNet-master/livermask')
    if not os.path.exists('./ckpt'):
        os.mkdir('./ckpt')

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
        num_workers = 4,
        pin_memory=True)

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
        train(train_loader, net, loss, epoch, optimizer, lr, batch_size=8)
        validate(val_loader, net, loss)

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
        ct = ct.view(-1, 512, 512)
        seg = seg.view(-1, 512, 512)
        for j in xrange(ct.shape[0]//batch_size):
            c = Variable(ct[j*batch_size:(j+1)*batch_size]).view(-1, 1, 512, 512).cuda()
            s = Variable(seg[j*batch_size:(j+1)*batch_size]).view(-1, 512, 512).cuda()
            if (s==0).all():
                del c, s
                continue
            out = net(c)
            loss_out = loss(out, s)
            optimizer.zero_grad()
            loss_out.backward()
            optimizer.step()
            losses += loss_out.data.cpu().numpy()
            del c, s, loss_out
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
    print('train loss %2.4f, time %2.4f' % (losses/131, et - st))

def validate(val_loader, net, loss):
    st = time.time()
    net.eval()
    losses = np.zeros(1)
    for i, (ct, seg) in enumerate(val_loader):
        ct = Variable(ct).cuda().view(-1, 512, 512)
        seg = Variable(seg).cuda().view(-1, 512, 512)
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

if __name__ == '__main__':
    main()
