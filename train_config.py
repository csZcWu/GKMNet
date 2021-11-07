# train_config
test_time = False
train = {}
train['train_img_path'] = '/home/lavie/zc/datasets/dd_dp_dataset_png/train_c/source'
train['train_gt_path'] = '/home/lavie/zc/datasets/dd_dp_dataset_png/train_c/target'
train['val_img_path'] = '/home/lavie/zc/datasets/dd_dp_dataset_png/val_c/source'
train['val_gt_path'] = '/home/lavie/zc/datasets/dd_dp_dataset_png/val_c/target'
train['test_img_path'] = '/home/lavie/zc/datasets/dd_dp_dataset_png/test_c/source'
train['test_gt_path'] = '/home/lavie/zc/datasets/dd_dp_dataset_png/test_c/target'
train['batch_size'] = 4
train['val_batch_size'] = 4
train['test_batch_size'] = 1
train['num_epochs'] = 4000
train['log_epoch'] = 1
train['optimizer'] = 'Adam'
train['learning_rate'] = 1e-6

# -- for SGD -- #
train['momentum'] = 0.9
train['nesterov'] = True

# config for save , log and resume
train['sub_dir'] = 'replicate'
train['resume'] = './save/replicate/1'
train['resume_epoch'] = 3818  # None means the last epoch
train['resume_optimizer'] = './save/replicate/1'

net = {}
net['xavier_init_all'] = True

loss = {}
loss['weight_l2_reg'] = 0.0
