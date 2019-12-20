# -*- coding: UTF-8 -*-
import pathlib
import torch
import torch.nn.functional as F
import importlib
import tempfile
import json
import shutil

def str2bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise RuntimeError('Boolean value expected')

def load_model(config):
    module = importlib.import_module('models.{}'.format(config['arch']))    # 得到文件及路径
    Network = getattr(module, 'Network')    # 初始化对应的model   然后读取module里面的Network类，但是不会去初始化
    return Network(config)  # 这里才会去初始化这个Network类

def onehot_encoding(label, n_classes):
    return torch.zeros(label.size(0), n_classes).to(label.device).scatter_(
        1, label.view(-1, 1), 1)

def cross_entropy_loss(input, target, reduction):
    logp = F.log_softmax(input, dim=1)
    loss = torch.sum(-logp * target, dim=1)     #  batch_size 个 loss
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(
            '`reduction` must be one of \'none\', \'mean\', or \'sum\'.')
    

def label_smoothing_criterion(epsilon, reduction):
    # 这里是一个闭包
    def _label_smoothing_criterion(preds, targets):
        n_classes = preds.size(1)
        device = preds.device
        
        onehot = onehot_encoding(targets, n_classes).float().to(device)     # 一个批次的标签转为one-hot
        targets = onehot * (1 - epsilon) + torch.ones_like(onehot).to(device) * epsilon / n_classes   # 滑动平均
        loss = cross_entropy_loss(preds, targets, reduction)
        return loss
        
    return _label_smoothing_criterion

def get_criterion(data_config):
    if data_config['use_label_smoothing']:
        train_criterion = label_smoothing_criterion(
            data_config['label_smoothing_epsilon'], reduction='mean')
    elif data_config['use_mixup']:
        pass
    else:
        train_criterion = torch.nn.CrossEntropyLoss()
    test_criterion = torch.nn.CrossEntropyLoss()
    return train_criterion, test_criterion

def _get_optimizer(model_parameters, optim_config):
    optimizer_name = optim_config['optimizer']  
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model_parameters,
            lr=optim_config['base_lr'],
            betas=optim_config['betas'],
            weight_decay=optim_config['weight_decay'])
    else:
        print("缺乏指定的优化器")
    return optimizer

def _get_scheduler(optimizer, optim_config):
    if optim_config['scheduler'] == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=optim_config['milestones'],
            gamma=optim_config['lr_decay'])
    else:
        scheduler = None
    return scheduler

def create_optimizer(model_parameters, optim_config):
    optimizer = _get_optimizer(model_parameters, optim_config)
    scheduler = _get_scheduler(optimizer, optim_config)
    return optimizer, scheduler

class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count
        
def accuracy(output, target, topk=(1, )):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1 / batch_size))
    return res

def save_epoch_logs(epoch_logs, outdir):
    dirname = outdir.resolve().as_posix().replace('/', '_')
    tempdir = pathlib.Path(tempfile.mkdtemp(prefix=dirname, dir='/tmp'))
    temppath = tempdir / 'log.json'
    with open(temppath, 'w') as fout:
        json.dump(epoch_logs, fout, indent=2)
    shutil.copy(temppath.as_posix(), outdir / temppath.name)
    shutil.rmtree(tempdir, ignore_errors=True)

def save_checkpoint(state, outdir):
    model_path = outdir / 'model_state.pth'
    best_model_path = outdir / 'model_best_state.pth'
    torch.save(state, model_path)
    if state['best_epoch'] == state['epoch']:
        shutil.copy(model_path, best_model_path)