# -*- coding: utf-8 -*-
import os
import sys
import time
import argparse
import numpy as np
import random  # 导入 random 模块
import pandas as pd  # 用于保存数据为 Excel 文件
import torch
from tqdm import tqdm
from accelerate import Accelerator  # 导入 accelerate 库
# ----------------------------------------
from MyValmore import val
from MyDataLoader import *
from alchemical_furnace1 import Network
# ----------------------------------------
from torch.utils.data import DataLoader, DistributedSampler
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from datetime import timedelta  # 确保导入 timedelta


def set_random_seed(seed):
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True  # 确保可重复性


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')  # 默认使用设备
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.002, help='initial learning rate')
    parser.add_argument('--n_steps', type=int, default=15, help='number of epochs to update learning rate')
    parser.add_argument('--n_epochs', type=int, default=30, help='number of epochs to train')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decaying factor')
    parser.add_argument('--trainset_dir', type=str, default='./Data/PVBLiF_NBU_9_9_32_32_patch/')
    return parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'


def init_distributed_mode():
    dist.init_process_group(backend='nccl')  # 使用 NCCL 后端
    if torch.cuda.is_available():
        torch.cuda.set_device(dist.get_rank())  # 设置当前进程的 GPU
    else:
        raise RuntimeError("CUDA is not available. Please check your environment.")


def train(train_loader, cfg, test_scene_id):

    if dist.get_rank() == 0:  # 仅主进程创建目录
        #os.makedirs('./logmore/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1]), exist_ok=True)
        dir_path = './logmore/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1])
        os.makedirs(dir_path, exist_ok=True)
        # 调试信息
        print(f'Directory created: {dir_path}' if os.path.exists(dir_path) else f'Failed to create: {dir_path}')
        print("现在是主进程创建目录 其他进程阻塞ing")
    dist.barrier()  # 等待所有进程同步

    sys.stdout = open('./logmore/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1])
                      + '/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1]) + '.txt', 'a')

    print(cfg)
    print(test_scene_id)

    accelerator = Accelerator()
    # 初始化模型
    net = Network().cuda()
    net.apply(weights_init_xavier)
    # 引入SyncBN，这句代码，会将普通BN替换成SyncBN。
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

    # Wrap the model in DDP with find_unused_parameters=True
    net = torch.nn.parallel.DistributedDataParallel(net, find_unused_parameters=True)

    optimizer = torch.optim.SGD(net.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=1e-4)
    net, optimizer = accelerator.prepare(net, optimizer)

    cudnn.benchmark = True
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)

    weights_list = []
    best_srcc = -float('inf')  # 初始化为负无穷
    best_epoch = 0
    best_model_path = ''

    for idx_epoch in range(cfg.n_epochs):
        loss_epoch = []
        start_time = time.time()
        train_loader.sampler.set_epoch(idx_epoch)

        for idx_iter, (data, score_label) in enumerate(
                tqdm(train_loader, desc=f'Epoch {idx_epoch + 1}/{cfg.n_epochs}', leave=False)):
            data, score_label = accelerator.prepare(data, score_label)

            score_label = score_label.view(score_label.size()[0], -1)
            data = torch.unsqueeze(data, dim=1)

            score_out, weight = net(data)
            score_label = score_label.to(score_out.device)

            weights_list.append(weight.detach().cpu().numpy())
            loss = torch.nn.MSELoss()(score_out, score_label)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            loss_epoch.append(loss.item())

        avg_loss = np.mean(loss_epoch)
        end_time = time.time()

        print('Test Epoch----%5d, loss---%f, Time---%f s, lr---%7f' % (
        idx_epoch + 1, avg_loss, end_time - start_time, scheduler.get_lr()[0]))

        # 仅主进程保存当前epoch的权重
        if accelerator.is_main_process:
            save_ckpt(accelerator, {'epoch': idx_epoch + 1, 'state_dict': net.state_dict(), 'loss': avg_loss},
                      save_path='./logmore/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1]) + '/',
                      filename='LF_epoch' + str(idx_epoch + 1) + '.pth.tar')
        dist.barrier()

        load_model_path = './logmore/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1]) + '/LF_epoch' + str(
            idx_epoch + 1) + '.pth.tar'
        val(valset_dir=cfg.trainset_dir, test_scene_id=test_scene_id, load_model_path=load_model_path)
        scheduler.step()

    # 关闭 stdout 重定向
    # sys.stdout.close()


def save_ckpt(accelerator, state, save_path='./logmore', filename='checkpoint.pth.tar'):
    # """保存模型检查点"""
    # if accelerator.is_main_process:  # 仅主进程保存
    os.makedirs(save_path, exist_ok=True)  # 创建保存路径（如果不存在）
    torch.save(state, os.path.join(save_path, filename))
    print(f"模型已保存到 {os.path.join(save_path, filename)}")


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)


def main(cfg):
    set_random_seed(42)  # 设置随机种子
    init_distributed_mode()  # 初始化分布式模式

    # 仅使用指定的场景对
    selected_scene_ids = [[0, 6], [0, 7], [0, 8], [0, 9], [0, 1]]  # 使用列表形式
    full_dataset_dir = cfg.trainset_dir

    for scene_id in selected_scene_ids:
        train_set = MyTrainSetLoader_Kfold(dataset_dir=full_dataset_dir, test_scene_id=scene_id)

        # 创建 DistributedSampler
        train_sampler = DistributedSampler(train_set)

        # 创建 DataLoader
        train_loader = DataLoader(dataset=train_set, sampler=train_sampler, batch_size=cfg.batch_size, num_workers=6,
                                  shuffle=False)
        # 训练
        train(train_loader, cfg, scene_id)

    # 结束 NCCL 进程组
    dist.destroy_process_group()


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)