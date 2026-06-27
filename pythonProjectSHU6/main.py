import sys
import time
import argparse
import numpy as np
import random  # 导入 random 模块
import torch
from tqdm import tqdm
# ----------------------------------------
from MyVal import val
from MyDataLoader import *
from alchemical_furnace2 import Network
# ----------------------------------------
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn


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

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.002, help='initial learning rate')
    parser.add_argument('--n_steps', type=int, default=55, help='number of epochs to update learning rate')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decaying factor')
    parser.add_argument('--trainset_dir', type=str,
                        default="/project/pythonProject3/Data/Sci_PVBLiF_SHU_9_9_32_32_patch/")

    return parser.parse_args()


def train(train_loader, cfg, test_scene_id):
    os.makedirs('./loglast/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1]), exist_ok=True)
    sys.stdout = open('./loglast/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1])
                      + '/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1]) + '.txt', 'a')
    print(cfg)
    print(test_scene_id)

    net = Network().to(cfg.device).apply(weights_init_xavier)  # 单卡训练：设置网络进行相应的初始化

    cudnn.benchmark = True
    optimizer = torch.optim.SGD([paras for paras in net.parameters() if paras.requires_grad == True],
                                lr=cfg.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)

    for idx_epoch in range(0, cfg.n_epochs):
        # 开始训练
        loss_epoch = []
        loss_list = []
        start_time = time.time()

        for idx_iter, (data, score_label) in enumerate(
                tqdm(train_loader, desc=f'Epoch {idx_epoch + 1}/{cfg.n_epochs}', leave=False)):
            data, score_label = Variable(data).to(cfg.device), Variable(score_label).to(cfg.device)
            score_label = score_label.view(score_label.size()[0], -1)
            # 将二维张量转化为一维标量
            data = torch.unsqueeze(data, dim=1)
            # 加一个维度（如通道维），使数据形状适合网络输入。
            # 原来的尺寸是（8，81，32，32）此时的data的尺寸是（8，1，81，32，32）

            score_out = net(data)
            loss = torch.nn.MSELoss().to(cfg.device)(score_out, score_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.data.cpu())

        loss_list.append(float(np.array(loss_epoch).mean()))
        end_time = time.time()
        print('Test Epoch----%5d,'
              ' loss---%f,'
              ' Time---%f s'
              ' lr---%7f s'
              % (idx_epoch + 1,
                 float(np.array(loss_epoch).mean()),
                 end_time - start_time,
                 scheduler.get_lr()[0]))

        save_ckpt({'epoch': idx_epoch + 1, 'state_dict': net.state_dict(), 'loss': loss_list, },
                  save_path='./loglast/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1]) + '/',
                  filename='PVBLiF_epoch' + str(idx_epoch + 1) + '.pth.tar')
        load_model_path = './loglast/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1]) + \
                          '/PVBLiF_epoch' + str(idx_epoch + 1) + '.pth.tar'
        val(valset_dir=cfg.trainset_dir, test_scene_id=test_scene_id, load_model_path=load_model_path)

        scheduler.step()


def save_ckpt(state, save_path='./loglast', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path, filename))


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)


def main(cfg):
    set_random_seed(42)  # 设置随机种子
    scene_num = 8
    selected_scene_ids = [[1, 2], [1, 4], [1,7],[2,5],[2,6],[4,5]]  # 使用列表形式
    full_dataset_dir = cfg.trainset_dir

    for scene_id in selected_scene_ids:
        test_scene_id = scene_id  # 在这里就指定了某些场景对是测试集 然后不选取这些进行训练
        # 这里会越界 只是测试02先
        train_set = MyTrainSetLoader_Kfold(dataset_dir=full_dataset_dir,
                                               test_scene_id=test_scene_id)  # MyTrainSetLoader_Kfold: 加载训练集，排除当前测试场景对。
        train_loader = DataLoader(dataset=train_set, num_workers=6, batch_size=cfg.batch_size, shuffle=True)
        # 具体来说，DataLoader会根据batch_size从train_set中提取多个连续索引（如0到7），并调用__getitem__多次来获取这些索引对应的数据和标签。
        train(train_loader, cfg, test_scene_id)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
