import torch
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        if not self.no_spatial:
            x_out = 1 / 3 * (self.hw(x) + self.cw(x.permute(0, 2, 1, 3).contiguous()).permute(0, 2, 1,
                                                                                              3).contiguous() + self.hc(
                x.permute(0, 3, 2, 1).contiguous()).permute(0, 3, 2, 1).contiguous())
        else:
            x_out = 1 / 2 * (self.cw(x.permute(0, 2, 1, 3).contiguous()).permute(0, 2, 1, 3).contiguous() + self.hc(
                x.permute(0, 3, 2, 1).contiguous()).permute(0, 3, 2, 1).contiguous())
        return x_out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class CLSTM_cell(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size

    """

    def __init__(self, input_chans, num_features, filter_size):
        super(CLSTM_cell, self).__init__()
        self.input_chans = input_chans
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2
        self.conv = nn.Conv2d(self.input_chans + self.num_features, 4 * self.num_features, self.filter_size, 1,
                              self.padding)

    def forward(self, input, hidden_state):
        hidden, c = hidden_state
        combined = torch.cat((input, hidden), 1)
        A = self.conv(combined)
        (ai, af, ao, ag) = torch.split(A, self.num_features, dim=1)
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)

        next_c = f * c + i * g
        next_h = o * torch.tanh(next_c)
        return next_h, next_c

    def init_hidden(self, batch_size, shape):
        return (torch.zeros(batch_size, self.num_features, shape[0], shape[1]).cuda(),
                torch.zeros(batch_size, self.num_features, shape[0], shape[1]).cuda())


class res_block(nn.Module):
    def __init__(self, ch_in):
        super(res_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True))

    def forward(self, x):
        y = x + self.conv(x)
        return y + self.conv1(y)


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.ta = TripletAttention()
        self.res_block = res_block(ch_out)

    def forward(self, x):
        return self.ta(self.res_block(self.conv(x)))


class conv_block_i(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_i, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.ta = TripletAttention()
        self.res_block = res_block(ch_out)

    def forward(self, x):
        return self.ta(self.res_block(self.conv(x)))


class conv_block1(nn.Module):
    def __init__(self, ch_in, ch_out, kernelsize=3):
        super(conv_block1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernelsize, stride=1, padding=int((kernelsize - 1) / 2), bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class conv_block_d(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.ta = TripletAttention()
        self.res_block = res_block(ch_out)

    def forward(self, x):
        return self.ta(self.res_block(self.conv(x)))


class conv_block_u(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_u, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.ta = TripletAttention()
        self.res_block = res_block(ch_out)

    def forward(self, x):
        return self.ta(self.res_block(self.conv(x)))


class SqueezeAttentionBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(SqueezeAttentionBlock, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = conv_block1(ch_in, ch_out)
        self.conv_atten = CLSTM_cell(ch_in, ch_out, 5)
        self.upsample = nn.Upsample(scale_factor=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden_state):
        x_res = self.conv(x)
        y = self.avg_pool(x)
        h, c = self.conv_atten(y, hidden_state)
        y = self.upsample(h)
        return self.sigmoid((y * x_res) + y) * 2 - 1, h, c


def gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return (gauss / gauss.sum()).cuda()


def gen_gaussian_kernel(window_size, sigma):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(_2D_window.expand(1, 1, window_size, window_size).contiguous())
    return window


class GaussianBlurLayer(nn.Module):
    def __init__(self, num_kernels=21, max_kernel_size=21, mode='TG', channels=3):
        super(GaussianBlurLayer, self).__init__()
        self.channels = channels
        kernel_size = 3
        weight = torch.zeros(num_kernels + 1, 1, max_kernel_size, max_kernel_size)
        for i in range(num_kernels):
            pad = int((max_kernel_size - kernel_size) / 2)
            weight[i + 1] = (F.pad(gen_gaussian_kernel(kernel_size, sigma=0.25 * (i + 1)).cuda(),
                                   [pad, pad, pad, pad])).squeeze(0)
            if i >= 2 and i % 2 == 0 and kernel_size < max_kernel_size:
                kernel_size += 2
        pad = int((max_kernel_size - 1) / 2)
        weight[0] = (F.pad(torch.FloatTensor([[[[1.]]]]).cuda(),
                           [pad, pad, pad, pad])).squeeze(0)

        kernel = np.repeat(weight, self.channels, axis=0).cuda()
        if mode == 'TG':
            self.weight = kernel
            self.weight.requires_grad = True
        elif mode == 'TR':
            self.weight = nn.Parameter(data=torch.randn(num_kernels * 3, 1, max_kernel_size, max_kernel_size),
                                       requires_grad=True)
        else:
            self.weight = kernel
            self.weight.requires_grad = False
        self.padding = int((max_kernel_size - 1) / 2)

    def __call__(self, x):
        # temp = self.weight.detach().unsqueeze(1).cpu().numpy()
        # for i in range(len(temp)//3):
        #     cv2.imwrite('kernels1/TG/' + str(i) + '.png', temp[i, 0, 0] * 255. * 1)
        x = F.conv2d(x, self.weight, padding=self.padding, groups=self.channels)
        return x


class SumLayer(nn.Module):
    def __init__(self, num_kernels=21, trainable=False):
        super(SumLayer, self).__init__()
        self.conv = nn.Conv2d(2 * (num_kernels + 1) * 3, 3, 1)

    def forward(self, x):
        return self.conv(x)


class MultiplyLayer1(nn.Module):
    def __init__(self):
        super(MultiplyLayer1, self).__init__()

    def forward(self, x, y):
        return x * torch.cat([y, y, y], dim=1)


class MultiplyLayer(nn.Module):
    def __init__(self):
        super(MultiplyLayer, self).__init__()
        self.ml = MultiplyLayer1()

    def forward(self, x, y):
        b, c, h, w = x.shape
        b1, c1, h1, w1 = y.shape
        return torch.cat([self.ml(x[:, :c // 2], y[:, :c1 // 2]), self.ml(x[:, c // 2:], y[:, c1 // 2:])], dim=1)


if __name__ == '__main__':
    ml = MultiplyLayer().cuda()
