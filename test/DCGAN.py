'''
@Project ：test 
@File    ：DCGAN.py
@Author  ：yuk
@Date    ：2024/3/4 9:41 
description：deep convolution GAN, refer: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
'''
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

manual_seed = 999
print("Random seed:", manual_seed)
random.seed(manual_seed)
# 设置随机数种子，让每次输出的结果一样，方便复现实验
torch.manual_seed(manual_seed)
# give same input, always return same output, otherwise error
torch.use_deterministic_algorithms(True)

data_root = r'E:\download\celeba'
# num_work
workers = 2
batch_size = 24
image_size = 64
# number of channels
in_channel = 3
# number of latend vector
nz = 100

ngf = 64
ndf = 64

num_epochs = 5
lr = 0.0002

ngpu = 0

trf = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]
)


# real_batch = next(iter(dataloader))
# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("traning Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
# plt.show()

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator net
class Generator(nn.Module):
    def __init__(self, ngpu):
        super().__init__()
        self.ngpu = ngpu
        self.gen_conv = nn.Sequential(
            nn.ConvTranspose2d(nz, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.GELU(),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 3, 1, 1, 0, bias=False),
            nn.Tanh()
        )
    def forward(self, input):
        return self.gen_conv(input)

# discriminator net
class Discirminator(nn.Module):
    def __init__(self, ngpu):
        super().__init__()
        self.ngpu = ngpu
        self.discr_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 3, 2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 3, 2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 3, 2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1024, 1, 1, padding=0, bias=False), # 1024x4x4
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(1024,1, 4, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.discr_conv(input)


if __name__ == '__main__':
    # use torchvision.datasets.imagefolder to load image
    dataset = dset.ImageFolder(root=data_root, transform=trf)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    netG = Generator(0).to(device=device)
    netD = Discirminator(0).to(device)
    # 多gpu处理
    if device.type == 'cuda' and ngpu > 0:
        netD = nn.DataParallel(netD, list(range(ngpu)))
        netG = nn.DataParallel(netG, list(range(ngpu)))
    netG.apply(weight_init)
    netD.apply(weight_init)
    # print(netG)
    # loss
    loss_func = nn.BCELoss()
    # optimizer
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

    # use test
    test_noise = torch.randn(16, nz, 1, 1, device=device)

    real_label = 1.
    fake_label = 0.

    # o = netD(data_noise)
    # print(o)
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    # training
    print('start training...')
    for epoch in range(num_epochs):
        # iter the datdaloader
        for i, (data, label) in enumerate(dataloader):
            real_data = data.to(device)
            b_size = real_data.size(0)
            r_label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # input real data to dis
            out_r = netD(real_data).view(-1)
            loss_D_r = loss_func(out_r, r_label)
            netD.zero_grad()
            loss_D_r.backward()

            avg_out_d_r = out_r.mean().item()
            # input fake data to dis
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            out_fake = netD(fake.detach()).view(-1)
            f_label = label.float().to(device)
            loss_D_f = loss_func(out_fake, f_label)
            loss_D_f.backward()
            avg_out_d_f = out_fake.mean().item()
            netD_loss = loss_D_r + loss_D_f
            # update D
            optimizerD.step()

            #Update G network: maximize log(D(G(z))),假数据给真标签
            netG.zero_grad()
            out = netD(fake).view(-1)
            loss_G = loss_func(out, r_label)
            loss_G.backward()
            avg_out_DGz = out.mean().item()
            optimizerG.step()
            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         netD_loss.item(), loss_G.item(), avg_out_d_r, avg_out_d_f, avg_out_DGz))

            # Save Losses for plotting later
            G_losses.append(loss_G.item())
            D_losses.append(netD_loss.item())
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(test_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

    # show loss
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # show G(x) generate data
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    HTML(ani.to_jshtml())






