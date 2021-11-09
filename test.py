import torch
import argparse
from network import GKMNet
from data import TestDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils import load_model, set_requires_grad, compute_psnr
from time import time
import os
import cv2
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import lpips
import train_config as config

loss_fn_alex = lpips.LPIPS(net='alex').cuda()

if __name__ == '__main__':
    dataset = TestDataset(config.train['test_img_path'], config.train['test_gt_path'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=1,
                            pin_memory=True)

    net = GKMNet().cuda()
    total = sum([param.nelement() for param in net.parameters()])

    print("Number of parameter: %.2fM" % (total / 1e6))
    set_requires_grad(net, False)
    max_psnr = 0
    e = 0
    last_epoch = load_model(net, config.train['resume'], epoch=2819)

    log_dir = 'test/{}'.format('DPD')
    os.system('mkdir -p {}'.format(log_dir))
    psnr_list = []
    ssim_list = []
    lpips_list = []

    total_time = 0
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            tt = time()
            for k in batch:
                batch[k] = batch[k].cuda(non_blocking=True)
                batch[k].requires_grad = False
            y256, _, _, t = net(batch['img256'], batch['img128'], batch['img64'])
            # cv2.imwrite('our_result/' + str(step) + '.png',
            #             cv2.cvtColor(y256[0].cpu().numpy().transpose([1, 2, 0]) * 255., cv2.COLOR_BGR2RGB))
            psnr_list.append(
                compute_psnr(torch.round(y256 * 255.), torch.round(batch['label256'] * 255.), 255).cpu().numpy())
            ssim_list.append(
                ssim(torch.round(y256 * 255.), torch.round(batch['label256'] * 255.), data_range=255,
                     size_average=False).cpu().numpy())
            lpips_list.append(loss_fn_alex(y256.cuda(), batch['label256']).cpu().numpy()[0][0][0][0])
            if step:
                total_time += (t - tt)
            if step % 100 == 100 - 1:
                t = time()
                psnr = np.mean(psnr_list)
                tqdm.write("{} / {} : psnr {} , {} img/s".format(step, len(dataloader) - 1, psnr,
                                                                 100 * 1 / (t - tt)))
                tt = t
    psnr = np.mean(psnr_list)
    print()
    print()
    print('psnr:', psnr)
    ssim = np.mean(ssim_list)
    print('ssim:', ssim)
    lpips_ = np.mean(lpips_list)
    print('lpips:', lpips_)
    print('avg_time:', total_time / 75.)
    with open('{}/psnr.txt'.format(log_dir), 'a') as log_fp:
        log_fp.write('epoch {} : psnr {}\n'.format(last_epoch, psnr))
