import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class ResBlock(nn.Module):
    def __init__(self, inChannels, midChannels ):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=midChannels, out_channels=midChannels, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=midChannels, out_channels=midChannels, kernel_size=(1, 3), padding=(0, 1))
        )
        self.conv1_1 = nn.Conv2d(in_channels=midChannels, out_channels=midChannels, kernel_size=1, padding=0)

        self.prelu = nn.PReLU()

        self.conv2 = nn.Conv2d(in_channels=midChannels*2, out_channels=midChannels, kernel_size=1, padding=0)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=midChannels, out_channels=midChannels, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=midChannels, out_channels=midChannels, kernel_size=(1, 3), padding=(0, 1))
        )


    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv1_1(x)
        outcat = torch.cat([out1, out2], 1)
        outcat = self.prelu(outcat)
        out3 = self.conv2(outcat)
        out3 = self.prelu(out3)
        out = self.conv3(out3)

        return out + x
    
class CropBlock(nn.Module):
    def __init__(self, inChannels, midChannels, kernel_size ,numResB, upscale_factor):
        super(CropBlock, self).__init__()
        self.crop_size = 32
        self.crop = transforms.CenterCrop(self.crop_size)
        self.crop_conv1 = nn.Conv2d(in_channels=inChannels, out_channels=midChannels, kernel_size=kernel_size, padding=kernel_size//2)
        self.crop_inResBlocks = nn.Sequential(*[ResBlock(midChannels,midChannels) for i in range(numResB)])
        self.crop_conv2 = nn.Conv2d(in_channels=midChannels, out_channels=upscale_factor**2, kernel_size=kernel_size, padding=kernel_size//2)

        self.prelu = nn.PReLU()


    def forward(self, x):
        B, C, H, W  = x.size()
        p = self.crop_size//2
        p2d = (W//2-p, W//2-p, H//2-p, H//2-p)
        cropimg = self.crop(x)
        cropout = self.crop_conv1(cropimg)
        cropout = self.crop_inResBlocks(cropout)
        cropout = self.prelu(cropout)
        cropout = self.crop_conv2(cropout)
        cropout = F.pad(cropout, p2d, mode='constant', value=0)

        return cropout

class BaselineModel(nn.Module):
    def __init__(self, inChannels=1, midChannels=64, numResB = 16, upscale_factor = 4):
        super(BaselineModel, self).__init__()
        kernel_size=3

        self.inCropBlock = CropBlock(inChannels=inChannels, midChannels=midChannels, kernel_size=kernel_size, numResB=numResB, upscale_factor=upscale_factor)
        self.conv1 = nn.Conv2d(in_channels=inChannels, out_channels=midChannels, kernel_size=kernel_size, padding=kernel_size//2)
        self.inResBlocks = nn.Sequential(*[ResBlock(midChannels,midChannels) for i in range(numResB)])
        self.conv2 = nn.Conv2d(in_channels=midChannels, out_channels=upscale_factor**2, kernel_size=kernel_size, padding=kernel_size//2)

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.prelu = nn.PReLU()

    def forward(self, x):
        cropout = self.inCropBlock(x)
        out1 = self.conv1(x)
        out1 = self.inResBlocks(out1)
        out1 = self.prelu(out1)
        out1 = self.conv2(out1)
        out1 = out1 + cropout
        out = self.pixel_shuffle(out1)

        return out