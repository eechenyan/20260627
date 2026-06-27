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
#----------------------------------------
from MyValmore import val
from MyDataLoader import *
from alchemical_furnace1 import Network
#----------------------------------------
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
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--n_steps', type=int, default=30, help='number of epochs to update learning rate')
    parser.add_argument('--n_epochs', type=int, default=31, help='number of epochs to train')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decaying factor')
    parser.add_argument('--trainset_dir', type=str, default='./Data/WithENP_PVBLiF_NBU_7_7_32_32_patch/')
    return parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

def init_distributed_mode(rank, world_size):
    #dist.init_process_group(backend='nccl')  # 使用 NCCL 后端
    dist.init_process_group(
        backend='nccl' if dist.is_nccl_available() else 'gloo',
        timeout=timedelta(seconds=7200000),  # 设置超时为7200000秒（2小时）
        rank=rank,
        world_size=world_size
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(dist.get_rank())  # 设置当前进程的 GPU
    else:
        raise RuntimeError("CUDA is not available. Please check your environment.")

def train(train_loader, cfg, test_scene_id):

    
    #if dist.get_rank() == 0:  # 仅主进程创建目录



    os.makedirs('./logmore/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1]), exist_ok=True)
    #调试信息
    print(f'Directory created: {dir_path}' if os.path.exists(dir_path) else f'Failed to create: {dir_path}')
    #print("现在是主进程创建目录 其他进程阻塞ing")
    #dist.barrier()  # 等待所有进程同步
    
    sys.stdout = open('./logmore/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1])
                      + '/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1]) + '.txt', 'a')

    

    print(cfg)
    print(test_scene_id)

 
    accelerator = Accelerator()
    # 初始化模型
    net = Network().cuda()
    net.apply(weights_init_xavier)

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

        for idx_iter, (data, score_label) in enumerate(tqdm(train_loader, desc=f'Epoch {idx_epoch + 1}/{cfg.n_epochs}', leave=False)):
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
        


        print('Test Epoch----%5d, loss---%f, Time---%f s, lr---%7f' % (idx_epoch + 1, avg_loss, end_time -                start_time, scheduler.get_lr()[0]))


        # 仅主进程保存当前epoch的权重   
        if accelerator.is_main_process: 
            save_ckpt(accelerator, {'epoch': idx_epoch + 1, 'state_dict': net.state_dict(), 'loss': avg_loss},
                      save_path='./logmore/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1]) + '/',
                      filename='LF_epoch' + str(idx_epoch + 1) + '.pth.tar')
        
        dist.barrier()
        

        # 仅主进程进行验证
        load_model_path = './logmore/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1]) + '/LF_epoch' + str(idx_epoch + 1) + '.pth.tar'
        #if accelerator.is_main_process:
        val(valset_dir=cfg.trainset_dir, test_scene_id=test_scene_id, load_model_path=load_model_path)
        #dist.barrier()  # 在关键点进行进程同步


            # 比较并保存SRCC最高的权重
        #if current_srcc > best_srcc:
            #best_srcc = current_srcc
            #best_epoch = idx_epoch + 1
            #best_model_path = load_model_path
            #save_ckpt(accelerator, {'epoch': best_epoch, 'state_dict': net.state_dict(), 'loss': avg_loss},
            #save_path='./logmore/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1]) + '/',
            #filename='best_LF_model.pth.tar')  # 使用固定的文件名
            #print(f'新最佳 SRCC: {best_srcc} 于 Epoch {best_epoch} 保存权重.')
            #dist.barrier()  # 在关键点进行进程同步

        #if (idx_epoch + 1) != cfg.n_epochs:
        #    os.remove(load_model_path)
        # 删除当前epoch的权重文件，除非是最后一个epoch或者是最佳权重文件

        scheduler.step()
    # 在所有 epoch 完成后，删除中间权重文件，保留最佳权重和最后一个epoch的权重文件
    #if idx_epoch + 1 == cfg.n_epochs:
        #for epoch in range(1, cfg.n_epochs):
            #epoch_model_path = f'./logmore/{test_scene_id[0]}_{test_scene_id[1]}/LF_epoch{epoch}.pth.tar'
            #if os.path.exists(epoch_model_path):
                #os.remove(epoch_model_path)

    # 训练结束后，将 weights_list 保存为 Excel 文件
    # weights_array = np.array(weights_list)
    # weights_df = pd.DataFrame(weights_array)
    # output_file_path = './logmore/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1]) + '/weights.xlsx'
    #if accelerator.is_main_process:
    # weights_df.to_excel(output_file_path, index=False, header=False)  # 保存到 Excel 文件
    #dist.barrier()  # 在关键点进行进程同步

    # 保存最佳的权重文件
    #if best_model_path:
        #print(f'保存最佳权重文件至: {best_model_path}')
        #dist.barrier()  # 在关键点进行进程同步
    
    # 关闭 stdout 重定向
    #sys.stdout.close()

def save_ckpt(accelerator, state, save_path='./logmore', filename='checkpoint.pth.tar'):
    #"""保存模型检查点"""
    #if accelerator.is_main_process:  # 仅主进程保存
    os.makedirs(save_path, exist_ok=True)  # 创建保存路径（如果不存在）
    torch.save(state, os.path.join(save_path, filename))
    print(f"模型已保存到 {os.path.join(save_path, filename)}")


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)

def main(cfg):
    set_random_seed(42)  # 设置随机种子
    RANK = int(os.environ['RANK'])  # 设置 RANK
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])  # 设置 WORLD_SIZE
    init_distributed_mode(RANK, WORLD_SIZE)  # 初始化分布式模式



    # 仅使用指定的场景对
    selected_scene_ids = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5]]  # 使用列表形式
    full_dataset_dir = cfg.trainset_dir

    for scene_id in selected_scene_ids:
        train_set = MyTrainSetLoader_Kfold(dataset_dir=full_dataset_dir, test_scene_id=scene_id)

        # 创建 DistributedSampler
        train_sampler = DistributedSampler(train_set)

        # 创建 DataLoader
        train_loader = DataLoader(dataset=train_set, sampler=train_sampler, batch_size=cfg.batch_size, num_workers=6, shuffle=False)
        #dist.barrier()
        # 训练
        train(train_loader, cfg, scene_id)


    # 结束 NCCL 进程组
    dist.destroy_process_group()

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)