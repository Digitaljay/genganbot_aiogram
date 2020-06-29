from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision


def image_loader(image_name,imsize,device):
    loader = transforms.Compose([
            transforms.Resize(imsize),  # нормируем размер изображения
            transforms.CenterCrop(imsize),
            transforms.ToTensor()])  # превращаем в удобный формат
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)



from torch.utils.data import DataLoader

from torch import nn
import torch
from torch.nn import functional as F
import functools
import itertools

class UnetGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1)
        self.pool0 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1,stride=2)  # 256 -> 128
        self.enc_conv1 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.pool1 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1,stride=2) # 128 -> 64
        self.enc_conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool2 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1,stride=2) # 64 -> 32
        self.enc_conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1)
        self.pool3 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1,stride=2) # 32 -> 16

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1)

        # decoder (upsampling)
        self.upsample0 = nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=2, stride=2, padding=0) # 16 -> 32
        self.dec_conv0 = nn.Conv2d(in_channels=512,out_channels=128,kernel_size=3,padding=1)
        self.upsample1 = nn.ConvTranspose2d(in_channels=128,out_channels=128,kernel_size=2, stride=2, padding=0) # 32 -> 64
        self.dec_conv1 = nn.Conv2d(in_channels=256,out_channels=64,kernel_size=3,padding=1)
        self.upsample2 = nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=2, stride=2, padding=0)  # 64 -> 128
        self.dec_conv2 = nn.Conv2d(in_channels=128,out_channels=32,kernel_size=3,padding=1)
        self.upsample3 = nn.ConvTranspose2d(in_channels=32,out_channels=32,kernel_size=2, stride=2, padding=0)  # 128 -> 256
        self.dec_conv3 = nn.Conv2d(in_channels=64,out_channels=3,kernel_size=3,padding=1)

    def forward(self, x):
        # encoder
        c0 = F.relu(self.enc_conv0(x))
        e0 = self.pool0(c0)
        c1 = F.relu(self.enc_conv1(e0))
        e1 = self.pool1(c1)
        c2 = F.relu(self.enc_conv2(e1))
        e2 = self.pool2(c2)
        c3 = F.relu(self.enc_conv3(e2))
        e3 = self.pool3(c3)

        # bottleneck
        cb=F.relu(self.bottleneck_conv(e3))
        b = self.upsample0(cb)

        # decoder
        conc0 = torch.cat([b, c3],dim=1)
        d0 = self.upsample1(F.relu(self.dec_conv0(conc0)))
        conc1 = torch.cat([d0, c2],dim=1)
        d1 = self.upsample2(F.relu(self.dec_conv1(conc1)))
        conc2 = torch.cat([d1, c1],dim=1)
        d2 = self.upsample3(F.relu(self.dec_conv2(conc2)))
        conc3 = torch.cat([d2, c0],dim=1)
        d3 = self.dec_conv3(conc3)  # no activation
        return d3
import numpy as np
class Gan():
    def __init__(self,img_given,img_res,imsize):
        self.img_given=img_given
        self.img_res=img_res
        self.imsize = imsize
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def paint(self):
        genB = UnetGenerator()
        genB.load_state_dict(torch.load("gog_genB20.pth"))

        photo=[image_loader(self.img_given,self.imsize,self.device)]
        photo_loader = DataLoader(photo, batch_size=1)
        a_real_test = Variable(iter(photo_loader).next()[0], requires_grad=True)
        a_real_test = a_real_test.to(self.device)

        genB.eval()

        with torch.no_grad():
            b_fake_test = genB(a_real_test)
        pic = b_fake_test.data
        print("got it")
        torchvision.utils.save_image(pic, self.img_res)
        print("saved as ",self.img_res)
        return self.img_res
