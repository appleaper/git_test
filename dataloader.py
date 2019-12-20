# -*- coding: UTF-8 -*-
import pathlib
import numpy as np
import torchvision
import transforms
import torch

class Dataset:
    def __init__(self, config):
        self.config = config    # 读入配置
        self.dataset_dir = 'imgs'   # 图片集
        self.real_dataset_dir = 'real_image'
        self._train_transforms = []     # 将所有的数据预处理的方法都添加到这里了
        self.train_transform = self._get_train_transform()  # 预处理转换
        self.test_transform = self._get_test_transform()

    def get_datasets(self):
        train_dataset = getattr(torchvision.datasets, self.config['dataset'])(
            self.dataset_dir,   # 数据集名称
            train=True,     # 是否训练
            transform=self.train_transform,     # 预处理
            download=True)      # 是否下载

        test_dataset = getattr(torchvision.datasets, self.config['dataset'])(
            self.dataset_dir,
            train=False,
            transform=self.test_transform,
            download=True)

        return train_dataset, test_dataset


    def _get_train_transform(self):
        if self.config['use_horizontal_flip']:   # 水平翻转
            self._add_horizontal_flip()
        self._add_normalization()  # 归一化
        self._add_to_tensor()
        return torchvision.transforms.Compose(self._train_transforms)

    def _get_test_transform(self):
        transform = torchvision.transforms.Compose([
            transforms.Normalize(self.mean, self.std),      # 标准化 保证和训练集数据区间相同
            transforms.ToTensor(),
        ])
        return transform

    def _add_horizontal_flip(self):
        self._train_transforms.append(
            torchvision.transforms.RandomHorizontalFlip())

    def _add_normalization(self):
        self._train_transforms.append(
            transforms.Normalize(self.mean, self.std))
    def _add_to_tensor(self):
        self._train_transforms.append(transforms.ToTensor())




class CIFAR(Dataset):   # 这里也初始化Dataset类  Dataset是CIFAR的父类,CIFAR继承父类Dataset
    # 字典加2个key
    __slots__ = ['mean','std']
    def __init__(self, config):     # 初始化方法
        self.size = 32
        if config['dataset'] == 'CIFAR10':
            # 这个均值和方差怎么来的呢？是自己求出来的，就是遍历图片，然后求出来
            self.mean = np.array([0.4914, 0.4822, 0.4465])  # 图像集中各个通道均值
            self.std = np.array([0.2470, 0.2435, 0.2616])   # 标准差
        elif config['dataset'] == 'CIFAR100':
            self.mean = np.array([0.5071, 0.4865, 0.4409])
            self.std = np.array([0.2673, 0.2564, 0.2762])
        super(CIFAR, self).__init__(config)     # 调用CIFAR的上级的init方法，并传入参数config

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def get_loader(config):
    batch_size = config['batch_size']   # batch_size
    num_workers = config['num_workers']     # 多线程
    use_gpu = config['use_gpu']     # 不使用GPU

    dataset_name = config['dataset']    # 读入数据集的名称
    if dataset_name in ['CIFAR10', 'CIFAR100']:     # 看看是那个数据集
        dataset = CIFAR(config)     # 将config放入CIFAR类中
    else:
        pass
    train_dataset, test_dataset = dataset.get_datasets()

    train_loader = torch.utils.data.DataLoader(
        train_dataset,              # 是一个对象
        batch_size=batch_size,      # 一个批次一个批次的取数据
        shuffle=True,               # 打乱顺序
        num_workers=num_workers,    # 多线程的读取
        pin_memory=use_gpu,         # 是不是用gpu的读取
        drop_last=True,             # 要是不够一个batch怎么办呢？直接丢弃了
        worker_init_fn=worker_init_fn,      # 这里是回调函数，初始化的时候就没有调用的
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,          # 测试集是不用打乱顺序的，因为只是测试一下准确率
        pin_memory=use_gpu,
        drop_last=False,        # 测试集每一张不都扔掉
    )

    return train_loader, test_loader