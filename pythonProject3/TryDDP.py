import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
from torch import distributed as dist
from torch.nn import parallel
from tqdm import tqdm
from MyVal import val
from MyDataLoader import *
from alchemical_furnace1 import Network
from torch.autograd import Variable
from torch.utils.data import DataLoader, DistributedSampler
import torch.backends.cudnn as cudnn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--n_steps', type=int, default=30, help='number of epochs to update learning rate')
    parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs to train')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decaying factor')
    parser.add_argument('--trainset_dir', type=str, default='./Data/PVBLiF_NBU_9_9_32_32_patch/')

    return parser.parse_args()


def train(train_loader, cfg, test_scene_id):
    # Initialize DDP
    if dist.get_rank() == 0:
        os.makedirs('./logmore/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1]), exist_ok=True)
        sys.stdout = open('./logmore/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1])
                          + '/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1]) + '.txt', 'a')

    print(cfg)
    print(test_scene_id)

    # Load the model
    net = Network().to(cfg.device).apply(weights_init_xavier)
    net = parallel.DistributedDataParallel(net, device_ids=[dist.get_rank()])

    cudnn.benchmark = True
    optimizer = torch.optim.SGD(net.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)

    weights_list = []

    for idx_epoch in range(cfg.n_epochs):
        loss_epoch = []
        start_time = time.time()

        for idx_iter, (data, score_label) in enumerate(
                tqdm(train_loader, desc=f'Epoch {idx_epoch + 1}/{cfg.n_epochs}', leave=False)):
            data, score_label = Variable(data).to(cfg.device), Variable(score_label).to(cfg.device)
            score_label = score_label.view(score_label.size()[0], -1)
            data = torch.unsqueeze(data, dim=1)

            score_out, weight = net(data)
            weights_list.append(weight.detach().cpu().numpy())
            loss = torch.nn.MSELoss().to(cfg.device)(score_out, score_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 汇总损失
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)  # 汇总所有进程的损失
            average_loss = loss.item() / dist.get_world_size()  # 计算平均损失
            loss_epoch.append(average_loss)

        # 计算并记录 epoch 平均损失
        epoch_loss = float(np.mean(loss_epoch))

        end_time = time.time()
        if dist.get_rank() == 0:  # 仅 rank 0 打印和保存
            print(
                f'Test Epoch----{idx_epoch + 1}, loss---{epoch_loss:.4f}, Time---{end_time - start_time:.2f}s, lr---{scheduler.get_lr()[0]:.7f}')

            save_ckpt({'epoch': idx_epoch + 1, 'state_dict': net.state_dict(), 'loss': epoch_loss},
                      save_path='./logmore/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1]) + '/',
                      filename='PVBLiF_epoch' + str(idx_epoch + 1) + '.pth.tar')

            load_model_path = './logmore/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1]) + \
                              '/PVBLiF_epoch' + str(idx_epoch + 1) + '.pth.tar'
            val(valset_dir=cfg.trainset_dir, test_scene_id=test_scene_id, load_model_path=load_model_path)

            if (idx_epoch + 1) != cfg.n_epochs:
                os.system('rm -r ./logmore/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1]) + \
                          'PVBLiF_epoch' + str(idx_epoch + 1) + '.pth.tar')

        scheduler.step()

    if dist.get_rank() == 0:  # 仅 rank 0 保存权重
        weights_array = np.array(weights_list)
        weights_df = pd.DataFrame(weights_array)
        output_file_path = './logmore/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1]) + '/weights.xlsx'
        weights_df.to_excel(output_file_path, index=False, header=False)


def save_ckpt(state, save_path='./logmore', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path, filename))


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)


def main(cfg):
    # 设置环境变量
    os.environ['MASTER_ADDR'] = '172.17.0.16'  # docker IP 地址
    os.environ['MASTER_PORT'] = '12345'
    os.environ["OMP_NUM_THREADS"] = "1"  # 根据需要修改线程数

    # 从环境变量读取
    cfg.world_size = int(os.environ['WORLD_SIZE'])
    cfg.rank = int(os.environ['RANK'])

    dist.init_process_group(backend='nccl', init_method='env://', world_size=cfg.world_size, rank=cfg.rank)

    scene_num = 14
    full_dataset_dir = cfg.trainset_dir
    for i in range(scene_num):
        for j in range(i + 1, scene_num):
            test_scene_id = [i, j]
            train_set = MyTrainSetLoader_Kfold(dataset_dir=full_dataset_dir, test_scene_id=test_scene_id)
            train_sampler = DistributedSampler(train_set, num_replicas=cfg.world_size, rank=cfg.rank, shuffle=True)
            train_loader = DataLoader(dataset=train_set, num_workers=6, batch_size=cfg.batch_size,
                                      sampler=train_sampler)

            train(train_loader, cfg, test_scene_id)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)

# torchrun --nproc_per_node=4 TryDDP.py