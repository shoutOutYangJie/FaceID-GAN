# import torch as t
# from torch.nn import Module
# from torch import nn
# from utils import initialize_weights


import torchvision.models as models
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import torchvision.models as models


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class G(nn.Module):
    def __init__(self, h, n, output_dim=(3, 64, 64)):
        super(G, self).__init__()
        self.n = n
        self.h = h

        channel, width, height = output_dim
        self.blocks = int(np.log2(width) - 2)

        print("[!] {} blocks in G ".format(self.blocks))

        self.fc = nn.Linear(h, 8 * 8 * n)

        conv_layers = []
        for i in range(self.blocks):
            conv_layers.append(nn.Conv2d(n, n, kernel_size=3, stride=1, padding=1))
            conv_layers.append(nn.ELU())
            conv_layers.append(nn.Conv2d(n, n, kernel_size=3, stride=1, padding=1))
            conv_layers.append(nn.ELU())

            if i < self.blocks - 1:
                conv_layers.append(nn.UpsamplingNearest2d(scale_factor=2))

        conv_layers.append(nn.Conv2d(n, channel, kernel_size=3, stride=1, padding=1))
        self.conv = nn.Sequential(*conv_layers)
        self.tanh = nn.Tanh()

        # self.tanh = nn.Tanh()

    def forward(self, x):
        fc_out = self.fc(x).view(-1, self.n, 8, 8)
        out = self.conv(fc_out)
        return self.tanh(out)

class D(nn.Module):
    def __init__(self, h, n, input_dim=(3, 64, 64)):
        super(D, self).__init__()

        self.n = n
        self.h = h

        channel, width, height = input_dim
        self.blocks = int(np.log2(width) - 2)

        print("[!] {} blocks in D ".format(self.blocks))

        encoder_layers = []
        encoder_layers.append(nn.Conv2d(3, n, kernel_size=3, stride=1, padding=1))

        prev_channel_size = n
        for i in range(self.blocks):
            channel_size = (i + 1) * n
            encoder_layers.append(nn.Conv2d(prev_channel_size, channel_size, kernel_size=3, stride=1, padding=1))
            encoder_layers.append(nn.ELU())
            encoder_layers.append(nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=1, padding=1))
            encoder_layers.append(nn.ELU())

            if i < self.blocks - 1:
                # Downsampling
                encoder_layers.append(nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=2, padding=1))
                encoder_layers.append(nn.ELU())

            prev_channel_size = channel_size

        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_encode = nn.Linear(8 * 8 * self.blocks * n, h)
        self.fc_decode = nn.Linear(h, 8 * 8 * n)

        decoder_layers = []
        for i in range(self.blocks):
            decoder_layers.append(nn.Conv2d(n, n, kernel_size=3, stride=1, padding=1))
            decoder_layers.append(nn.ELU())
            decoder_layers.append(nn.Conv2d(n, n, kernel_size=3, stride=1, padding=1))
            decoder_layers.append(nn.ELU())

            if i < self.blocks - 1:
                decoder_layers.append(nn.UpsamplingNearest2d(scale_factor=2))

        decoder_layers.append(nn.Conv2d(n, channel, kernel_size=3, stride=1, padding=1))
        self.decoder = nn.Sequential(*decoder_layers)
        # self.tanh = nn.Tanh()

    def forward(self, x):
        #   encoder        [ flatten ]
        x = self.encoder(x).view(x.size(0), -1)
        # print(x)
        x = self.fc_encode(x)

        #   decoder
        x = self.fc_decode(x).view(-1, self.n, 8, 8)
        x = self.decoder(x)

        return x


#
# class G(Module):
#     def __init__(self,hidden_num,repeat_num):
#         super(G,self).__init__()
#         self.hidden_num = hidden_num
#         self.fc1 = nn.Linear(64,8*8*hidden_num)
#         self._make_layers(repeat_num=repeat_num,hidden_num=hidden_num)
#         self.out_layer = nn.Conv2d(hidden_num,3,3,1,padding=1)
#         initialize_weights(self)
#     def _make_layers(self,repeat_num,hidden_num):
#         self.layers = nn.Sequential()
#         for idx in range(repeat_num):
#             layers = [
#                 nn.Conv2d(hidden_num,hidden_num,3,1,padding=1),
#                 nn.ELU(),
#                 nn.Conv2d(hidden_num,hidden_num,3,1,padding=1),
#                 nn.ELU()
#             ]
#             self.layers.add_module('layers_%d'%(idx+1),nn.Sequential(*layers))
#             if idx <repeat_num-1:
#                 up_x = nn.UpsamplingNearest2d(scale_factor=2)
#                 self.layers.add_module(name='up_%d'%(idx+1),module=up_x)
#
#     def forward(self, x):
#         out = self.fc1(x)
#         out = out.view(-1,self.hidden_num,8,8)
#         out = self.layers(out)
#         out = self.out_layer(out)
#         return out
#
#
# class D(Module):
#     def __init__(self,hidden_num,repeat_num):
#         super(D,self).__init__()
#         self.hidden_num = hidden_num
#         self.repeat_num = repeat_num
#         #encoder
#         self.conv1 = nn.Conv2d(3,hidden_num,3,1,padding=1)
#         self._make_EN_layers(hidden_num,repeat_num)
#         self.fc1 = nn.Linear(8*8*(repeat_num+1)*hidden_num,hidden_num)
#
#         #decoder
#         self.fc2 = nn.Linear(hidden_num,8*8*hidden_num)
#         self._make_DE_layers(hidden_num,repeat_num)
#         self.out_layer = nn.Conv2d(hidden_num, 3, 3, 1, padding=1)
#
#         initialize_weights(self)
#
#     def _make_EN_layers(self,hidden_num,repeat_num):
#         self.layers_en = nn.Sequential()
#         for idx in range(repeat_num):
#             prev_channel_num = hidden_num*(1+idx)
#             channel_num = hidden_num*(2+idx)
#             layers = [
#                 nn.Conv2d(prev_channel_num,prev_channel_num,3,1,padding=1),
#                 nn.ELU(),
#                 nn.Conv2d(prev_channel_num,channel_num,3,1,padding=1),
#                 nn.ELU()
#             ]
#             self.layers_en.add_module('layers_%d'%(idx+1),nn.Sequential(*layers))
#             if idx < repeat_num-1:
#                 sub_x = nn.Conv2d(channel_num,channel_num,3,2,padding=1)
#                 self.layers_en.add_module(name='sub_%d'%(idx+1),module=sub_x)
#
#     def _make_DE_layers(self,hidden_num,repeat_num):
#         self.DE_layyers = nn.Sequential()
#         for idx in range(repeat_num):
#             layers = [
#                 nn.Conv2d(hidden_num,hidden_num,3,1,padding=1),
#                 nn.ELU(),
#                 nn.Conv2d(hidden_num,hidden_num,3,1,padding=1),
#                 nn.ELU()
#             ]
#             self.DE_layyers.add_module('layers_%d'%(idx+1),nn.Sequential(*layers))
#             if idx < repeat_num-1:
#                 up_x = nn.UpsamplingNearest2d(scale_factor=2)
#                 self.DE_layyers.add_module('up_%d'%(idx+1),up_x)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.layers_en(out) #8 384 8 8
#         out = out.view(-1,8*8*(self.repeat_num+1)*self.hidden_num)
#         out = self.fc1(out)
#         out = self.fc2(out)
#         out = out.view(-1,self.hidden_num,8,8)
#         out = self.DE_layyers(out)
#         out = self.out_layer(out)
#         return out

if __name__=='__main__':
    # z = t.randn(8,64)
    g = G(hidden_num=64,repeat_num=3)
    d = D(hidden_num=64,repeat_num=3)
    print(list(g.children()))
    print(list(d.children()))