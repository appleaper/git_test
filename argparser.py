# -*- coding: UTF-8 -*-
import json
from collections import OrderedDict
import torch

def _set_default_values(args):
    '''加载默认参数，要是一开始没有设置，就使用默认默认参数'''
    if args.config is not None:
        with open(args.config, 'r') as fin:     # 读取默认的json
            config = json.load(fin)     # 加载json文件，返回字典

        d_args = vars(args)     
        for config_key, default_config in config.items():   # 遍历默认json
            for key, default_value in default_config.items():   # 打开具体内容
                # 要是外面没有指定，那么就用里面的默认值替换它
                if key not in d_args or d_args[key] is None:   
                    setattr(args, key, default_value)   # 设置一下

    return args

def _cleanup_args(args):
    if args.use_horizontal_flip is None:
        if args.dataset in ['CIFAR10', 'CIFAR100', 'FashionMNIST']:
            args.use_horizontal_flip = True
    if not args.use_label_smoothing:
        args.label_smoothing_epsilon = None
    if args.dataset == 'CIFAR10':
        args.input_shape = (args.batch_size, 3, 32, 32)     # 这里改为batch_size,
        args.n_classes = 10
    return args

def _get_model_config(args):
    keys = [
        'arch',
        'input_shape',
        'n_classes',
        # vgg
        'n_channels',
        'n_layers',
        'use_bn',
    ]
    config = _args2config(args, keys)
    return config

def _check_optim_config(config):
    optimizer = config['optimizer']
    for key in ['base_lr', 'weight_decay']:
        message = 'Key `{}` 必须指定.'.format(key)
        assert key in config.keys(), message
    if optimizer == 'adam':
        for key in ['betas']:
            message = '使用 Adam, key `{}` 必须指定.'.format(
                key)
            assert key in config.keys(), message
    scheduler = config['scheduler']
    if scheduler == 'multistep':
        for key in ['milestones', 'lr_decay']:
            message = 'Key `{}` must be specified.'.format(key)
            assert key in config.keys(), message
        
    

def _get_optim_config(args):
    keys = [
        'epochs',
        'batch_size',
        'optimizer',
        'base_lr',
        'weight_decay',
        'no_weight_decay_on_bn',
        'gradient_clip',
        'scheduler',
        'milestones',
        'lr_decay',
        'betas',
    ]
    json_keys = ['milestones', 'betas']
    config = _args2config(args, keys)

    _check_optim_config(config)

    return config

def _args2config(args, keys):
    args = vars(args)  # 所有配置变为字典
    config = OrderedDict()
    for key in keys:    # 获取所有args的key
        value = args.get(key, None)      # 获取key的值

        if value is None:        # 要是key的值是None就退出当前循环
            continue
        config[key] = value      # 把配置的键和值都赋值给config字典

    return config

def _check_data_config(config):
    if config['use_cutout'] and config['use_dual_cutout']:
        raise ValueError(
            'Only one of `use_cutout` and `use_dual_cutout` can be `True`.')
    if sum([
            config['use_mixup'], config['use_ricap'], config['use_dual_cutout']
    ]) > 1:
        raise ValueError(
            'Only one of `use_mixup`, `use_ricap` and `use_dual_cutout` can be `True`.'
        )

def _get_data_config(args):
    keys = [
        'dataset',
        'n_classes',
        'num_workers',
        'batch_size',
        'use_horizontal_flip',
        'use_label_smoothing',
        'label_smoothing_epsilon',
        'reset_data',
        'use_random_crop',
        'use_cutout',
        'use_dual_cutout',
        'use_random_erasing',
        'use_mixup',
        'use_ricap',
    ]
    config = _args2config(args, keys)
    config['use_gpu'] = args.device != 'cpu'    # 选定操作设备是cpu
    _check_data_config(config)
    return config
    
def _get_run_config(args):
    keys = [
        'outdir',
        'seed',
        'test_first',
        'device',
        'fp16',
        'use_amp',
        'tensorboard',
        'tensorboard_train_images',
        'tensorboard_test_images',
        'tensorboard_model_params',
    ]
    config = _args2config(args, keys)

    return config

def get_config(args):
    if args.config is None:
        args.config = 'configs/{}.json'.format(args.arch)
        
    args = _set_default_values(args)    # 设置默认值
    args = _cleanup_args(args)  # 清理一些参数
    config = OrderedDict({      # 将参数整理分类
        'model_config': _get_model_config(args),
        'optim_config': _get_optim_config(args),
        'data_config': _get_data_config(args),
        'run_config': _get_run_config(args),    # 运行参数
        # 'env_info': _get_env_info(args),
    })
    return config