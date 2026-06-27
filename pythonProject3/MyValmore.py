import os
import torch
from MyDataLoader import *
from alchemical_furnace1 import Network
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from scipy.stats import spearmanr as SRCC
import h5py
import numpy as np
import os

def val(valset_dir, test_scene_id, load_model_path, type='val'):
    device = torch.cuda.current_device()  # 获取当前进程的设备
    net = Network().to(device)
    cudnn.benchmark = True
    
    # 允许使用 NumPy 的一些全局对象
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    
    
    # 加载模型
    print("加载模型...")
    try:
      #print("当前 NCCL 超时设置为:", os.environ.get('NCCL_TIMEOUT'))
      model = torch.load(load_model_path, map_location='cuda:{}'.format(device))
    except Exception as e:
      print(f"加载模型时出错: {e}")
      return None
    #model = torch.load(load_model_path, map_location='cuda:{}'.format(device))
    
    # 移除前缀并加载状态字典
    state_dict = model['state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')  # 移除 'module.' 前缀
        new_state_dict[new_key] = value
    
    net.load_state_dict(new_state_dict)
    net.eval()
    print("模型加载成功，开始验证...")

    label_list = []
    data_list = []
    scene_list = ['I01R0', 'I02R0', 'I03R0', 'I04R0', 'I05R0', 'I06R0', 'I07R0',
                  'I08R0', 'I09R0', 'I10R0', 'I11R0', 'I12R0', 'I13R0', 'I14R0']  # 14个场景

    for scene_index, test_scene in enumerate(test_scene_id):
        print(f"正在处理场景: {scene_list[test_scene]} (场景 {scene_index + 1}/{len(test_scene_id)})")
        image_path = os.path.join(valset_dir, scene_list[test_scene])
        image_list = os.listdir(image_path)

        for image_index, test_image in enumerate(image_list):
            patch_path = os.path.join(image_path, test_image)
            patch_list = os.listdir(patch_path)
            output_list = 0
            
            for val_patch_index, val_patch in enumerate(patch_list):
                each_patch_path = os.path.join(patch_path, val_patch)
                with h5py.File(each_patch_path, 'r') as hf:
                    label = np.array(hf.get('score_label'))
                    data = np.array(hf.get('dis_data'))
                    data = data / 255
                    data = np.expand_dims(data, axis=0)
                    data = np.expand_dims(data, axis=0)
                    data = torch.from_numpy(data.copy())
                    data = Variable(data).to(device)

                with torch.no_grad():
                    out_score, weight = net(data)
                output_list += out_score.detach().cpu().numpy().item()
                
            label_list.append(label.item())
            data_list.append(output_list / len(patch_list))

    loss = torch.nn.MSELoss().to(device)(torch.tensor(data_list).to(device), torch.tensor(label_list).to(device))
    val_SRCC = SRCC(data_list, label_list).correlation
    print(f"{type} SRCC :----    {val_SRCC:.6f}")
    print(f"{type} LOSS :----    {loss:.6f}")
    return val_SRCC

if __name__ == '__main__':
    # 设置模型文件路径和其他参数
    load_model_path = "/project/pythonProject3/log10/0_1/PVBLiF_epoch10.pth.tar"  # 替换为实际模型路径
    valset_dir = './Data/PVBLiF_NBU_9_9_32_32_patch/'  # 替换为实际数据路径
    test_scene_id = [0, 1]  # 示例场景ID

    # 调用验证函数
    val(valset_dir, test_scene_id, load_model_path, type='validation')
    
    
