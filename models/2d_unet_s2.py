import torch
from torch import nn

class UNet_2d(nn.Module):
    def __init__(self):
        super(UNet_2d, self).__init__()

        forw_chs = [3, 64, 128, 256, 512, 1024]
        back_chs = [1024, 512, 256, 128]
        for i in xrange(len(forw_chs) - 1):
            block = nn.Sequential(
                    nn.Conv2d(forw_chs[i], forw_chs[i+1], (3, 3), padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(forw_chs[i+1], forw_chs[i+1], (3, 3), padding=1),
                    nn.ReLU(inplace=True))
            setattr(self, 'encoder'+str(i+1), block)
        for i in xrange(len(back_chs) - 1):
            block = nn.Sequential(
                    nn.Conv2d(back_chs[i], back_chs[i+1], (3, 3), padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(back_chs[i+1], back_chs[i+1], (3, 3), padding=1),
                    nn.ReLU(inplace=True))
            setattr(self, 'decoder'+str(i+1), block)
        self.decoder4 = nn.Sequential(
                    nn.Conv2d(128, 64, (3, 3), padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, (3, 3), padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 1, (1, 1), padding=0))
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.upconv = nn.Upsample(scale_factor=(2, 2))
        self.upconv1 = nn.ConvTranspose2d(1024, 512, 2, stride=2, padding=0)
        self.upconv2 = nn.ConvTranspose2d(512, 256, 2, stride=2, padding=0)
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2, padding=0)
        self.upconv4 = nn.ConvTranspose2d(128, 64, 2, stride=2, padding=0)

    def forward(self, x):
        out1 = self.encoder1(x)
        out2 = self.encoder2(self.maxpool(out1))
        out3 = self.encoder3(self.maxpool(out2))
        out4 = self.encoder4(self.maxpool(out3))
        out5 = self.encoder5(self.maxpool(out4))
        out6 = self.decoder1(torch.cat([out4, self.upconv1(out5)], 1))
        out7 = self.decoder2(torch.cat([out3, self.upconv2(out6)], 1))
        out8 = self.decoder3(torch.cat([out2, self.upconv3(out7)], 1))
        out9 = self.decoder4(torch.cat([out1, self.upconv4(out8)], 1))
        return out9

def get_model():
    net = UNet_2d()
    return net
    
