import torch
from torch import nn
import model.common as common
import torch.nn.functional as F
import matplotlib.pyplot as plt
from moco.builder import MoCo
import numpy as np

def make_model(args):
    return BlindSR(args)


# --------------------------自定义网络模块------------------------------- #

class MSRB_Block(nn.Module):
    def __init__(self):
        super(MSRB_Block, self).__init__()

        self.conv_1_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_1_2 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_3_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_3_2 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_5_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv_5_2 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, stride=1, padding=2, bias=True)
        self.confusion = nn.Conv2d(in_channels=576, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity_data = x
        output_1_1 = self.relu(self.conv_1_1(x))#[8,64,256,256]=>[8,64,256,256]
        output_3_1 = self.relu(self.conv_3_1(x))#[8,64,256,256]=>[8,64,256,256]
        output_5_1 = self.relu(self.conv_5_1(x))#[8,64,256,256]=>[8,64,256,256]
        input_2 = torch.cat([output_1_1, output_3_1, output_5_1], 1)#[8,128,256,256]
        output_1_2 = self.relu(self.conv_1_2(input_2))#[8,192,256,256]=>[8,192,256,256]
        output_3_2 = self.relu(self.conv_3_2(input_2))#[8,128,256,256]=>[8,128,256,256]
        output_5_2 = self.relu(self.conv_5_2(input_2))#[8,128,256,256]=>[8,128,256,256]
        output = torch.cat([output_1_2, output_3_2, output_5_2], 1)#[8,256,256,256]
        output = self.confusion(output)#[8,256,256,256]==>#[8,64,256,256]
        output = torch.add(output, identity_data)#[8,64,256,256]=>[8,64,256,256]
        return output

class DA_conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, reduction):
        super(DA_conv, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size

        self.kernel = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64, 64 * self.kernel_size * self.kernel_size, bias=False)
        )
        self.conv = common.default_conv(channels_in, channels_out, 1)
        self.ca = CA_layer(channels_in, channels_out, reduction)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        b, c, h, w = x[0].size()

        # branch 1
        kernel = self.kernel(x[1]).view(-1, 1, self.kernel_size, self.kernel_size)#torch.Size([512, 1, 3, 3])
        out = self.relu(F.conv2d(x[0].view(1, -1, h, w), kernel, groups=b*c, padding=(self.kernel_size-1)//2))#==》torch.Size([1, 512, 256, 256])
        out = self.conv(out.view(b, -1, h, w))#[8,64,256,256]

        # branch 2
        out = out + self.ca(x)  #[8,64,256,256] + [8,64,256,256] = [8,64,256,256]

        return out

### modify
class DA_CBAM_conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, reduction):
        super(DA_CBAM_conv, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size

        self.kernel = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64, 64 * self.kernel_size * self.kernel_size, bias=False)
        )
        self.conv = common.default_conv(channels_in, channels_out, 1)
        self.ca = CA_layer(channels_in, channels_out, reduction)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        b, c, h, w = x[0].size()

        # branch 1
        out = self.ca(x)#[8,64,256,256]

        # branch 2
        kernel = self.kernel(x[1]).view(-1, 1, self.kernel_size, self.kernel_size)#torch.Size([512, 1, 3, 3])
        out = self.relu(F.conv2d(out.view(1, -1, h, w), kernel, groups=b*c, padding=(self.kernel_size-1)//2))#==》torch.Size([1, 512, 256, 256])
        out = self.conv(out.view(b, -1, h, w))#[8,64,256,256]

        return out


class CA_layer(nn.Module):
    def __init__(self, channels_in, channels_out, reduction):
        super(CA_layer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels_in, channels_in//reduction, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_in // reduction, channels_out, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        att = self.conv_du(x[1][:, :, None, None])#[8, 64]==>torch.Size([8, 64, 1, 1])

        return x[0] * att   #[8, 64, 256, 256]*[8, 64, 1, 1] = [8, 64, 256, 256]


class DAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction):
        super(DAB, self).__init__()

        # self.da_conv1 = DA_conv(n_feat, n_feat, kernel_size, reduction)
        # self.da_conv2 = DA_conv(n_feat, n_feat, kernel_size, reduction)
        self.da_conv1 = DA_CBAM_conv(n_feat, n_feat, kernel_size, reduction)
        self.da_conv2 = DA_CBAM_conv(n_feat, n_feat, kernel_size, reduction)
        self.conv1 = conv(n_feat, n_feat, kernel_size)
        self.conv2 = conv(n_feat, n_feat, kernel_size)
        ###
        self.MSRB_block = MSRB_Block()

        self.relu =  nn.LeakyReLU(0.1, True)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''

        out = self.relu(self.da_conv1(x))#[8,64,256,256]
        out = self.relu(self.conv1(out))#[8,64,256,256]
        out = self.relu(self.da_conv2([out, x[1]]))#[8,64,256,256]
        out = self.conv2(out) + x[0]#[8,64,256,256]
        ###
        out = self.MSRB_block(out)#[8,64,256,256]


        return out#[8,64,256,256]

        # plt.figure();plt.imshow(out.squeeze(0))


class DAG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_blocks):
        super(DAG, self).__init__()
        self.n_blocks = n_blocks
        modules_body = [
            DAB(conv, n_feat, kernel_size, reduction) \
            for _ in range(n_blocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        res = x[0]
        for i in range(self.n_blocks):
            res = self.body[i]([res, x[1]])
        res = self.body[-1](res)#Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        res = res + x[0]

        return res


class DASR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(DASR, self).__init__()

        self.n_groups = 5
        n_blocks = 5
        n_feats = 64
        kernel_size = 3
        reduction = 8
        # scale = 4
        scale = int(args.scale[0])

        # RGB mean for DIV2K
        # rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_mean = (0.39779539, 0.40924516, 0.36850663)
        # AID
        # [0.39779539 0.40924516 0.36850663]
        rgb_std = (1.0, 1.0, 1.0)
        # ### UCMerced
        # rgb_mean = (0.480996, 0.487935, 0.448912)

        self.sub_mean = common.MeanShift(255.0, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(255.0, rgb_mean, rgb_std, 1)

        # head module
        modules_head = [conv(3, n_feats, kernel_size)]
        self.head = nn.Sequential(*modules_head)

        # compress
        self.compress = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.LeakyReLU(0.1, True)
        )

        # body
        modules_body = [
            DAG(common.default_conv, n_feats, kernel_size, reduction, n_blocks) \
            for _ in range(self.n_groups)
        ]
        # modules_body.append(conv(n_feats, n_feats, kernel_size))
        ###
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

        # tail
        modules_tail = [common.Upsampler(conv, scale, n_feats, act=False),
                        conv(n_feats, 3, kernel_size)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x, k_v):
        k_v = self.compress(k_v)

        # sub mean
        x = self.sub_mean(x)

        # head
        x = self.head(x)

        # body
        res = x
        ###
        out = res
        for i in range(self.n_groups):
            res = self.body[i]([res, k_v])
            # res = torch.add(res, x)
            ###
            # out = torch.cat([out, res], 1)

        res = self.body[-1](res)
        ###
        # res = self.body[-1](out)
        res = res + x

        # tail
        x = self.tail(res)

        # add mean
        x = self.add_mean(x)

        return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.E = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        fea = self.E(x).squeeze(-1).squeeze(-1)
        out = self.mlp(fea)

        return fea, out


class BlindSR(nn.Module):
    def __init__(self, args):
        super(BlindSR, self).__init__()

        # Generator
        self.G = DASR(args)

        # Encoder
        self.E = MoCo(base_encoder=Encoder)

    def forward(self, x):
        if self.training:
            x_query = x[:, 0, ...]                          # b, c, h, w  x=[8,64,256,256]==>x_query=[8,256,256]
            x_key = x[:, 1, ...]                            # b, c, h, w    ==>x_key=[8,256,256]

            # degradation-aware represenetion learning
            fea, logits, labels = self.E(x_query, x_key)

            # degradation-aware SR
            sr = self.G(x_query, fea)

            return sr, logits, labels
        else:
            # degradation-aware represenetion learning
            fea = self.E(x, x)

            # degradation-aware SR
            sr = self.G(x, fea)

            return sr

def main():
    pass
    input = [torch.Tensor(8, 8, 64, 256, 256), torch.Tensor(8, 64)]
    # CA_l = CA_layer(64, 64, 2)
    # output = CA_l(input)    #torch.Size([8, 64, 256, 256])
    # DA_c = DA_CBAM_conv(64, 64, 3, 2)
    # output = DA_c(input)#torch.Size([8, 64, 256, 256])
    MoCo_TEST = MoCo(base_encoder=Encoder)
    x_query = input[0][:, 0, ...]
    x_key = input[0][:, 1, ...]
    fea, logits, labels = MoCo_TEST(x_query, x_key)

    # DAb = DAB(common.default_conv, 64, 3, 2)
    # output = DAb(input)#output.shape torch.Size([8, 64, 256, 256])
    # DAg = DAG(common.default_conv, 64, 3, 2, 5)
    # output = DAg(input)#output.shape torch.Size([8, 64, 256, 256])
    # DAS = DASR().cuda()
    # input_x_q, input_x_f = torch.Tensor(2, 3, 128, 128).cuda(), torch.Tensor(2, 256).cuda()
    # output = DAS(input_x_q, input_x_f)#output.shape torch.Size([8, 3, 512, 512])

if __name__ == '__main__':
    main()
    # from torchsummary import summary
    # from thop import profile
    #
    # macs, params = profile(DASR().to(device), inputs=(torch.Tensor(1, 3, 256, 256).cuda(), torch.Tensor(1, 256).cuda()))
    # print('Total macc:{}, Total params: {}'.format(macs, params))