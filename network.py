from layers import *
import torch
from time import time
import torch.nn.functional as F


class GKMNet(nn.Module):
    def __init__(self, kernel_mode='FG', num_gaussian_kernels=21, gaussian_kernel_size=21):
        super(type(self), self).__init__()
        super().__init__()
        self.GCM = GaussianBlurLayer(num_gaussian_kernels, gaussian_kernel_size, kernel_mode)
        '''backbone'''
        self.in_block = conv_block_i(3, 32)
        self.conv_block1_1 = conv_block_d(32, 64)
        self.conv_block2_1 = conv_block_d(64, 128)
        self.bottle_neck_1 = conv_block(128, 128)
        self.conv_block3_1 = conv_block_u(128, 64)
        self.conv_block4_1 = conv_block_u(64, 32)
        '''APU'''
        self.APU = SqueezeAttentionBlock(32, 2 * (num_gaussian_kernels + 1))
        '''entry-wise multiplication'''
        self.MultiplyLayer = MultiplyLayer()
        '''summation'''
        self.SumLayer = SumLayer(num_gaussian_kernels)

    def forward_step(self, input_blurry, last_output, hidden_state):
        ''' Gaussian Reblurring '''
        gy = self.GCM(input_blurry)
        gx = self.GCM(last_output)

        '''Feature Extraction'''
        f_i = self.in_block(input_blurry)
        f1_1 = self.conv_block1_1(f_i)
        f2_1 = self.conv_block2_1(f1_1)
        bn_1 = self.bottle_neck_1(f2_1)
        f3_1 = self.conv_block3_1(bn_1 + f2_1)
        f4_1 = self.conv_block4_1(f3_1 + f1_1)

        '''Weight Maps Generation'''
        weights, h, c = self.APU(f4_1, hidden_state)

        '''Entry-wise Multiplication and Summation'''
        result = self.SumLayer(self.MultiplyLayer(torch.cat([gy, gx], dim=1), weights))
        return result, h, c

    def forward(self, input_blur_256, input_blur_128, input_blur_64):
        h, c = self.APU.conv_atten.init_hidden(
            input_blur_64.shape[0],
            (input_blur_64.shape[-2] // 2, input_blur_64.shape[-1] // 2))
        """The forward process"""
        '''scale 1'''
        db64, h, c = self.forward_step(input_blur_64, input_blur_64, (h, c))
        h = F.upsample(h, scale_factor=2, mode='bilinear')
        c = F.upsample(c, scale_factor=2, mode='bilinear')
        '''scale 2'''
        db128, h, c = self.forward_step(input_blur_128, F.upsample(db64, scale_factor=2, mode='bilinear'), (h, c))
        h = F.upsample(h, scale_factor=2, mode='bilinear')
        c = F.upsample(c, scale_factor=2, mode='bilinear')
        '''scale 3'''
        db256, _, _ = self.forward_step(input_blur_256, F.upsample(db128, scale_factor=2, mode='bilinear'), (h, c))
        return db256, db128, db64, time()
