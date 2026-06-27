import sys
import time
import argparse
import numpy as np
import pandas as pd
import os
import torch
from tqdm import tqdm
from MyVal import val
from MyDataLoader import *
from alchemical_furnace1 import Network
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import deepspeed
import os
os.environ["MASTER_PORT"] = "29501"  # 设置为一个可用的端口

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_steps', type=int, default=15)
    parser.add_argument('--n_epochs', type=int, default=3)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--trainset_dir', type=str, default='./Data/PVBLiF_NBU_9_9_32_32_patch/')
    parser.add_argument('--deepspeed_config', type=str, default='ds_config.json')  # DeepSpeed配置文件
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    return parser.parse_args()

def train(train_loader, cfg, test_scene_id, model, optimizer, scheduler):
    os.makedirs('./logmoreds/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1]), exist_ok=True)
    sys.stdout = open('./logmoreds/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1]) + '/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1]) + '.txt', 'a')
    print(cfg)
    print(test_scene_id)

    cudnn.benchmark = True
    weights_list = []

    for idx_epoch in range(cfg.n_epochs):
        loss_epoch = []
        start_time = time.time()

        for idx_iter, (data, score_label) in enumerate(tqdm(train_loader, desc=f'Epoch {idx_epoch + 1}/{cfg.n_epochs}', leave=False)):
            data, score_label = data.to(cfg.device), score_label.to(cfg.device)
            score_label = score_label.view(score_label.size()[0], -1)
            data = torch.unsqueeze(data, dim=1)

            # Forward pass
            score_out, weight = model(data)

            # Compute loss
            loss = torch.nn.MSELoss()(score_out, score_label)
            weights_list.append(weight.detach().cpu().numpy())

            # Backward pass and optimization
            model.backward(loss)
            model.step()

            loss_epoch.append(loss.item())

        end_time = time.time()
        print('Test Epoch----%5d, loss---%f, Time---%f s' % (idx_epoch + 1, np.mean(loss_epoch), end_time - start_time))

        # Save checkpoint
        save_ckpt({'epoch': idx_epoch + 1, 'state_dict': model.state_dict(), 'loss': loss_epoch},
                  save_path='./logmoreds/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1]) + '/',
                  filename='PVBLiF_epoch' + str(idx_epoch + 1) + '.pth.tar')

        # Call validation
        load_model_path = './logmoreds/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1]) + '/PVBLiF_epoch' + str(idx_epoch + 1) + '.pth.tar'
        val(valset_dir=cfg.trainset_dir, test_scene_id=test_scene_id, load_model_path=load_model_path)

        scheduler.step()

    # Save weights to Excel
    weights_array = np.array(weights_list)
    weights_df = pd.DataFrame(weights_array)
    output_file_path = './logmoreds/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1]) + '/weights.xlsx'
    weights_df.to_excel(output_file_path, index=False, header=False)

def save_ckpt(state, save_path='./logmoreds', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path, filename))

def weights_init_xavier(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data)

def main(cfg):
    scene_num = 14
    full_dataset_dir = cfg.trainset_dir

    # Initialize DeepSpeed
    model = Network()
    model.apply(weights_init_xavier)

    # Prepare DeepSpeed
    model, optimizer, _, scheduler = deepspeed.initialize(args=cfg, model=model, model_parameters=model.parameters())

    for i in range(scene_num):
        for j in range(i + 1, scene_num):
            test_scene_id = [i, j]
            train_set = MyTrainSetLoader_Kfold(dataset_dir=full_dataset_dir, test_scene_id=test_scene_id)
            train_loader = DataLoader(dataset=train_set, num_workers=6, batch_size=cfg.batch_size, shuffle=True)
            train(train_loader, cfg, test_scene_id, model, optimizer, scheduler)

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)