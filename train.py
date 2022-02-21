import sys

import lpips
from pytorch_msssim import ssim
from tqdm import tqdm

import train_config as config
from data import Dataset, TestDataset
from log import TensorBoardX
from network import GKMNet
from utils import *

# from time import time

log10 = np.log(10)
MAX_DIFF = 1
le = 1


def worker_init_fn_seed(worker_id):
    seed = 10
    seed += worker_id
    np.random.seed(seed)


def compute_loss(db256, db128, db64, batch, epoch):
    assert db256.shape[0] == batch['label256'].shape[0]
    global le
    temp = (epoch % 1000) % 200
    if 0 <= temp < 100 and epoch >= 1000:
        le = 100
        print('ssim', end='')
        '''use ssim loss'''
        loss = 0
        loss += mse(db256, batch['label256'])
        psnr = 10 * torch.log(MAX_DIFF ** 2 / loss) / log10
        loss = 0
        loss += 1 - torch.mean(ssim(db256, batch['label256'], data_range=1, size_average=False))
        loss += 1 - torch.mean(ssim(db128, batch['label128'], data_range=1, size_average=False))
        loss += 1 - torch.mean(ssim(db64, batch['label64'], data_range=1, size_average=False))

    else:
        le = 1
        print('mse', end='')
        '''use mse loss'''
        loss = 0
        loss += mse(db256, batch['label256'])
        psnr = 10 * torch.log(MAX_DIFF ** 2 / loss) / log10
        loss += mse(db128, batch['label128'])
        loss += mse(db64, batch['label64'])
    if epoch < 1000:
        le = 1000

    return {'mse': loss, 'psnr': psnr}


def backward(loss, optimizer):
    optimizer.zero_grad()
    loss['mse'].backward()
    optimizer.step()
    return


def set_learning_rate(optimizer, epoch):
    if epoch < 2000:
        optimizer.param_groups[0]['lr'] = config.train['learning_rate']
    else:
        optimizer.param_groups[0]['lr'] = config.train['learning_rate'] * 0.1


if __name__ == "__main__":
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    tb = TensorBoardX(config_filename='train_config.py', sub_dir=config.train['sub_dir'])
    log_file = open('{}/{}'.format(tb.path, 'train.log'), 'w')

    train_dataset = Dataset(config.train['train_img_path'], config.train['train_gt_path'])
    test_dataset = TestDataset(config.train['test_img_path'], config.train['test_gt_path'])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train['batch_size'], shuffle=True,
                                                   drop_last=True, num_workers=8, pin_memory=True,
                                                   worker_init_fn=worker_init_fn_seed)
    val_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.train['test_batch_size'],
                                                 shuffle=False,
                                                 drop_last=True, num_workers=1, pin_memory=True)

    mse = torch.nn.MSELoss().cuda()
    mae = torch.nn.L1Loss().cuda()

    net = torch.nn.DataParallel(GKMNet()).cuda()
    total = sum([param.nelement() for param in net.parameters()])

    print("Number of parameter: %.2fM" % (total / 1e6))

    assert config.train['optimizer'] in ['Adam', 'SGD']
    if config.train['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=config.train['learning_rate'],
                                     weight_decay=config.loss['weight_l2_reg'])
    if config.train['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=config.train['learning_rate'],
                                    weight_decay=config.loss['weight_l2_reg'], momentum=config.train['momentum'],
                                    nesterov=config.train['nesterov'])

    last_epoch = -1

    if config.train['resume'] is not None:
        last_epoch = load_model(net, config.train['resume'], epoch=config.train['resume_epoch'])

    if config.train['resume_optimizer'] is not None:
        _ = load_optimizer(optimizer, net, config.train['resume_optimizer'], epoch=config.train['resume_epoch'])
        assert last_epoch == _

    # train_loss_epoch_list = []

    train_loss_log_list = []
    val_loss_log_list = []
    first_val = True

    t = time.time()
    best_val_psnr = 0
    best_net = None
    best_optimizer = None
    for epoch in tqdm(range(last_epoch + 1, config.train['num_epochs']), file=sys.stdout):
        set_learning_rate(optimizer, epoch)
        tb.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch * len(train_dataloader), 'train')
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), file=sys.stdout,
                                desc='training'):
            t_list = []
            for k in batch:
                batch[k] = batch[k].cuda(non_blocking=True)
                batch[k].requires_grad = False
            # cv2.imshow('target', batch['label256'][0].detach().cpu().numpy().transpose([1, 2, 0]))
            t = time.time()
            db256, db128, db64, _ = net(batch['img256'], batch['img128'], batch['img64'])
            loss = compute_loss(db256, db128, db64, batch, epoch)
            backward(loss, optimizer)

            for k in loss:
                loss[k] = float(loss[k].cpu().detach().numpy())
            train_loss_log_list.append({k: loss[k] for k in loss})
            for k, v in loss.items():
                tb.add_scalar(k, v, epoch * len(train_dataloader) + step, 'train')

        # test and log
        if first_val or epoch % le == le - 1:
            with torch.no_grad():
                first_val = False
                psnr_list = []
                ssim_list = []
                lpips_list = []
                total_time = 0
                for step, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader), file=sys.stdout,
                                        desc='validating'):
                    for k in batch:
                        batch[k] = batch[k].cuda(non_blocking=True)
                        batch[k].requires_grad = False
                    # cv2.imshow('target', batch['label256'][0].detach().cpu().numpy().transpose([1, 2, 0]))
                    # db256, _, _, t = net(batch['img256'], batch['img128'], batch['img64'])
                    tt = time.time()
                    db256, db128, db64, t = net(batch['img256'], batch['img128'], batch['img64'])
                    loss = compute_loss(db256, db128, db64, batch, epoch)
                    for k in loss:
                        loss[k] = float(loss[k].cpu().detach().numpy())
                    # cv2.imwrite('our_result/' + str(step) + '.png',
                    #             cv2.cvtColor(y256[0].cpu().numpy().transpose([1, 2, 0]) * 255., cv2.COLOR_BGR2RGB))
                    psnr_list.append(compute_psnr(db256, batch['label256'], 1).cpu().numpy())
                    ssim_list.append(ssim(db256, batch['label256'], data_range=1, size_average=False).cpu().numpy())
                    lpips_list.append(loss_fn_alex(db256, batch['label256']).cpu().numpy()[0][0][0][0])
                    if step:
                        total_time += (t - tt)
                    for k in loss:
                        loss[k] = float(loss[k])
                    val_loss_log_list.append({k: loss[k] for k in loss})
                train_loss_log_dict = {k: float(np.mean([dic[k] for dic in train_loss_log_list])) for k in
                                       train_loss_log_list[0]}
                val_loss_log_dict = {k: float(np.mean([dic[k] for dic in val_loss_log_list])) for k in
                                     val_loss_log_list[0]}
                for k, v in val_loss_log_dict.items():
                    tb.add_scalar(k, v, (epoch + 1) * len(train_dataloader), 'val')
                if best_val_psnr < val_loss_log_dict['psnr']:
                    best_val_psnr = val_loss_log_dict['psnr']
                    save_model(net, tb.path, epoch)
                    save_optimizer(optimizer, net, tb.path, epoch)
                elif epoch % 50 == 0:
                    save_model(net, tb.path, epoch)
                    save_optimizer(optimizer, net, tb.path, epoch)
                train_loss_log_list.clear()
                val_loss_log_list.clear()
                tt = time.time()
                log_msg = ""
                log_msg += "epoch {} , {:.2f} imgs/s".format(epoch, (
                        config.train['log_epoch'] * len(train_dataloader) * config.train['batch_size'] + len(
                    val_dataloader) * config.train['val_batch_size']) / (tt - t))

                log_msg += " | train : "
                for idx, k_v in enumerate(train_loss_log_dict.items()):
                    k, v = k_v
                    if k == 'acc':
                        log_msg += "{} {:.3%} {}".format(k, v, ',')
                    else:
                        log_msg += "{} {:.5f} {}".format(k, v, ',')
                log_msg += "  | val : "
                for idx, k_v in enumerate(val_loss_log_dict.items()):
                    k, v = k_v
                    if k == 'acc':
                        log_msg += "{} {:.3%} {}".format(k, v, ',')
                    else:
                        log_msg += "{} {:.5f} {}".format(k, v, ',' if idx < len(val_loss_log_list) - 1 else '')
                tqdm.write(log_msg, file=sys.stdout)
                sys.stdout.flush()
                log_file.write(log_msg + '\n')
                log_file.flush()
                t = time.time()
