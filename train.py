# -*- coding: UTF-8 -*-
import collections
import argparse
import json,time
import logging
from argparser import get_config
from utils import str2bool,AverageMeter
import pathlib
import numpy as np
import random
import torch
from dataloader import get_loader
import utils
import torchvision




logging.basicConfig(
    format='[%(asctime)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)
global_step = 0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', '-a', default='vgg', type=str)    # 网络名称
    parser.add_argument('--config',type=str)    # 直接外部配置文件
    parser.add_argument('--device', type=str, default='cpu')       # 是否用gpu
    parser.add_argument(
        '--optimizer', type=str, choices=['sgd', 'adam', 'lars'],default='adam')    # 优化器
    parser.add_argument('--betas', type=str,default=[0.9,0.999])
    parser.add_argument(
        '--dataset',
        type=str,
        default='CIFAR10',
        choices=['CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST', 'KMNIST', 'XRay'])     # 数据集
    parser.add_argument('--use_horizontal_flip', type=str2bool)     # 是不是水平翻转
    parser.add_argument(
        '--use_label_smoothing', action='store_true', default=True)     # 标签平滑
    parser.add_argument('--label_smoothing_epsilon', type=float, default=0.1)
    parser.add_argument('--no_weight_decay_on_bn', action='store_true')
    parser.add_argument('--gradient_clip', type=float, default=5.0)  # 梯度裁剪，防止梯度爆炸，爆炸的话模型不稳定震荡。
    parser.add_argument('--num_workers', type=int, default=0)  # 多线程
    parser.add_argument('--reset_data', default=True, type=bool)
    parser.add_argument('--use_random_crop', type=str2bool, default=False)  # 是不是用随机裁剪 default=True
    parser.add_argument('--use_cutout', action='store_true', default=False)  # 应该是随机遮挡，也是一种增强  default=True
    parser.add_argument(
        '--use_dual_cutout', action='store_true', default=False)
    parser.add_argument(
        '--use_random_erasing', action='store_true', default=False)      # 随机擦除 default=True
    parser.add_argument('--use_mixup', action='store_true', default=False)  # 混合增强 default=True
    parser.add_argument('--use_ricap', action='store_true', default=False)  # 增强的一种

    parser.add_argument('--outdir', type=str, default='results/vgg/00')  # 输出路径
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--test_first', type=str2bool, default=False)  # 要不要先测试一下
    parser.add_argument('--fp16', default=False, type=bool)  # 普通是f32转f16进行参数压缩
    parser.add_argument('--use_amp', action='store_true')  # amp
    parser.add_argument('--batch_size', type=int, default=256)  # 一个batch放8张图片



    args = parser.parse_args()  # 参数解析
    config = get_config(args)
    return config

def train(epoch, model, optimizer, scheduler, criterion, train_loader, config):
    global global_step
    
    run_config = config['run_config']
    optim_config = config['optim_config']
    data_config = config['data_config']
    device = torch.device(run_config['device'])

    logger.info('Train epoch {}'.format(epoch))  # 记录一下当前的epoch
    
    model.train()
    
    loss_meter = AverageMeter()     # 初始化这个类
    accuracy_meter = AverageMeter()
    start = time.time()
    for step, (data, targets) in enumerate(train_loader):
        global_step += 1
        targets = targets.type(torch.LongTensor)    # 转为LongTensor格式
        if torch.cuda.device_count() == 1:
            data = data.to(device)

        if data_config['use_mixup']:
            t1, t2, lam = targets
            targets = (t1.to(device), t2.to(device), lam)
        else:
            targets = targets.to(device)
        optimizer.zero_grad()   # 即将梯度初始化为零

        if 'ghost_batch_size' not in optim_config.keys():
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()     # 根据损失更新模型
            
        if 'gradient_clip' in optim_config.keys():      # 梯度截断
            torch.nn.utils.clip_grad_norm_(model.parameters(),optim_config['gradient_clip'])
        optimizer.step()
        if optim_config['scheduler'] in ['multistep', 'sgdr']:
            scheduler.step(epoch - 1)
        loss_ = loss.item()
        num = data.size(0)
        accuracy = utils.accuracy(outputs, targets)[0].item()   # 计算准确率
        loss_meter.update(loss_, num)       # 每个样本的平均损失
        accuracy_meter.update(accuracy, num)    # 平均准确率把？

        if step % 10 == 0:
            logger.info('Epoch {} Step {}/{} '
                        'Loss {:.4f} ({:.4f}) '
                        'Accuracy {:.4f} ({:.4f})'.format(
                            epoch,
                            step,
                            len(train_loader),
                            loss_meter.val,
                            loss_meter.avg,
                            accuracy_meter.val,
                            accuracy_meter.avg,
                        ))
    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))
    
    train_log = collections.OrderedDict({
        'epoch':
        epoch,
        'train':
        collections.OrderedDict({
            'loss': loss_meter.avg,
            'accuracy': accuracy_meter.avg,
            'time': elapsed,
        }),
    })
    return train_log

def test(epoch, model, criterion, test_loader, run_config):
    logger.info('Test {}'.format(epoch))
    device = torch.device(run_config['device'])
    model.eval()
    loss_meter = AverageMeter()
    correct_meter = AverageMeter()
    start = time.time()
    with torch.no_grad():
        for step, (data, targets) in enumerate(test_loader):
            data = data.to(device)
            targets = targets.to(device)
            targets = targets.type(torch.LongTensor)
            outputs = model(data)
            loss = criterion(outputs, targets)

            _, preds = torch.max(outputs, dim=1)

            loss_ = loss.item()
            correct_ = preds.eq(targets).sum().item()
            num = data.size(0)

            loss_meter.update(loss_, num)
            correct_meter.update(correct_, 1)

        accuracy = correct_meter.sum / len(test_loader.dataset)

        logger.info('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(
            epoch, loss_meter.avg, accuracy))

        elapsed = time.time() - start
        logger.info('Elapsed {:.2f}'.format(elapsed))

        test_log = collections.OrderedDict({
            'epoch':
                epoch,
            'test':
                collections.OrderedDict({
                    'loss': loss_meter.avg,
                    'accuracy': accuracy,
                    'time': elapsed,
                }),
        })
        return test_log

def update_state(state, epoch, accuracy, model, optimizer):
    state['state_dict'] = model.state_dict()
    state['optimizer'] = optimizer.state_dict()
    state['epoch'] = epoch
    state['accuracy'] = accuracy

    # update best accuracy
    if accuracy > state['best_accuracy']:
        state['best_accuracy'] = accuracy
        state['best_epoch'] = epoch

    return state

def main():
    # parse command line argument and generate config dictionary
    config = parse_args()   # 解析参数
    logger.info(json.dumps(config, indent=2))   # 打印一下参数，作为日志
    run_config = config['run_config']
    optim_config = config['optim_config']

    # set random seed
    seed = run_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    epoch_seeds = np.random.randint(
        np.iinfo(np.int32).max // 2, size=optim_config['epochs'])

    outdir = pathlib.Path(run_config['outdir'])
    outdir.mkdir(exist_ok=True, parents=True)

    # save config as json file in output directory
    outpath = outdir / 'config.json'
    with open(outpath, 'w') as fout:
        json.dump(config, fout, indent=2)

    train_loader, test_loader = get_loader(config['data_config'])  # 读入了训练集和测试集
    
    logger.info('Loading model...')     # 加载模型文字描述记录到日志中
    model = utils.load_model(config['model_config'])    # 工具类，加载模型
    n_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    logger.info('n_params: {}'.format(n_params))  # 这个可能是计算参数量是多少？

    device = torch.device(run_config['device'])     # 指定是cpu还是GPU
    if device.type == 'cuda' and torch.cuda.device_count() > 1:     # 假如有GPU
        model = torch.nn.DataParallel(model)
    model.to(device)
    logger.info('模型加载到设备完成')

    train_criterion, test_criterion = utils.get_criterion(config['data_config'])
    logger.info('增强完成')

    # 挑选出要更新的参数
    params = filter(lambda x: x.requires_grad, model.parameters())

    optim_config['steps_per_epoch'] = len(train_loader)     #多少个训练样本
    optimizer, scheduler = utils.create_optimizer(params, optim_config)     # 获取模型的操作函数，还是变化学习率
    
    state = {
        'config': config,
        'state_dict': None,
        'optimizer': None,
        'epoch': 0,
        'accuracy': 0,
        'best_accuracy': 0,
        'best_epoch': 0,
    }
    
    epoch_logs = []
    for epoch, seed in zip(range(1, optim_config['epochs'] + 1), epoch_seeds):
        np.random.seed(seed)
        train_log = train(epoch, model, optimizer, scheduler, train_criterion,
                          train_loader, config)
        test_log = test(epoch, model, test_criterion, test_loader, run_config)
        epoch_log = train_log.copy()
        epoch_log.update(test_log)
        epoch_logs.append(epoch_log)
        utils.save_epoch_logs(epoch_logs, outdir)

        # update state dictionary
        state = update_state(state, epoch, epoch_log['test']['accuracy'],
                             model, optimizer)

        # save model
        utils.save_checkpoint(state, outdir)

        
    

if __name__ == '__main__':
    main()