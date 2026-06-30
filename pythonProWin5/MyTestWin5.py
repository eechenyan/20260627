import torch
from MyDataLoader import *
import numpy as np
import os  # 新增：用于路径处理
import argparse  # 新增：用于接收命令行参数
from alchemical_furnace7 import Network
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from scipy.stats import spearmanr as SROCC
from logistic import verify_performance
import h5py  # 补充：代码中使用了h5py但未显式导入


def test_model(load_all_model_path, valset_dir):  # 新增：接收外部路径参数
    ### Win5 
    # 注释掉原来的硬编码路径，改为通过参数传入
    # load_all_model_path = '/project/pythonProject3/Netss/Win5Nets/'
    # valset_dir = '/project/pythonProject3/Data/Sci_PVBLiF_Win5_9_9_32_32_patch/'
    dataset_name = 'Win5'
    scene_list = ['Bikes', 'dishes', 'Flowers', 'greek', 'museum', 'Palais_du_Luxembourg', 'rosemary', 'Sphynx',
                   'Swans_1', 'Vespa']
    test_scene_num = 2
    distorted_num = 22
    scene_num = 10

    # 确保结果存储路径存在（改为相对路径，避免依赖系统目录）
    results_dir = './Results/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # 自动选择可用设备（避免依赖固定GPU编号，增强兼容性）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = Network().to(device)
    cudnn.benchmark = True

    # 检查传入的权重路径是否有效
    if not os.path.exists(load_all_model_path):
        raise ValueError(f"权重路径不存在：{load_all_model_path}")
    all_model = os.listdir(load_all_model_path)

    label_list = np.zeros([test_scene_num * distorted_num, len(all_model)])
    data_list = np.zeros([test_scene_num * distorted_num, len(all_model)])
    val_SRCC_all = []
    val_SRCCme_all = []
    val_KROCC_all = []
    val_PLCC_all = []
    val_RMSE_all = []
    test_scene_id_list = [[0,2],[8,9],[2,5],[0,7],[3,8],[4,9],[5,7],[5,9],[6,8],[6,9]]

    test_count = 1
    for id, model_name in enumerate(test_scene_id_list):
        # 构建模型路径（使用os.path.join增强路径兼容性）
        load_model_path = os.path.join(
            load_all_model_path, 
            f"{model_name[0]}_{model_name[1]}", 
            "DAFLFQA.pth.tar"
        )
        # 检查模型文件是否存在
        if not os.path.exists(load_model_path):
            raise FileNotFoundError(f"模型文件不存在：{load_model_path}")

        # 加载模型
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        model = torch.load(load_model_path, map_location=device, weights_only=False)
        net.load_state_dict(model['state_dict'], strict=False)
        net.eval()

        index = 0
        test_scene_id = [int(model_name[0]), int(model_name[1])]

        for test_scene in test_scene_id:
            # 构建图像路径（使用os.path.join适配不同系统）
            image_path = os.path.join(valset_dir, scene_list[test_scene])
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"场景路径不存在：{image_path}")
            
            image_list = os.listdir(image_path)
            for test_image in image_list:
                patch_path = os.path.join(image_path, test_image)
                patch_list = os.listdir(patch_path)
                output_list = []
                VS_list = []

                for val_patch in patch_list:
                    if int(val_patch[:-3]) < int(len(patch_list) / 30):
                        continue
                    each_patch_path = os.path.join(patch_path, val_patch)
                    
                    with h5py.File(each_patch_path, 'r') as hf:
                        label = np.array(hf.get('score_label'))
                        data = np.array(hf.get('dis_data'))
                        data = data / 255
                        data = np.expand_dims(data, axis=0)
                        data = np.expand_dims(data, axis=0)
                        data = torch.from_numpy(data.copy())
                        data = Variable(data).to(device)
                        VS = np.array(hf.get('SCI'))
                    
                    with torch.no_grad():
                        out_score = net(data)
                    output_list.append(out_score.cpu().numpy().item() * VS.item())
                    VS_list.append(VS.item())

                label_list[index, id] = label.item()
                data_list[index, id] = sum(output_list) / sum(VS_list)
                index += 1

        # 计算评估指标
        val_SRCC = SROCC(data_list[:,id], label_list[:,id]).correlation
        srocc, krocc, plcc, rmse = verify_performance(label_list[:, id], data_list[:, id])
        val_SRCCme_all.append(srocc)
        val_KROCC_all.append(krocc)
        val_PLCC_all.append(plcc)
        val_RMSE_all.append(rmse)

        print(f'Results of {test_count}-th test:')
        print('SROCC :----    %f' % val_SRCC)
        print(f"SROCC: {srocc}, KROCC: {krocc}, PLCC: {plcc}, RMSE: {rmse}")
        val_SRCC_all.append(val_SRCC)
        test_count += 1

    # 输出平均指标
    print('Average SROCC :----   %f' % np.mean(val_SRCC_all))
    print('Average SROCCme :----   %f' % np.mean(val_SRCCme_all))
    print('Average KROCC :----   %f' % np.mean(val_KROCC_all))
    print('Average PLCC :----   %f' % np.mean(val_PLCC_all))
    print('Average RMSE :----   %f' % np.mean(val_RMSE_all))

    # 保存结果（使用相对路径）
    results_file = os.path.join(results_dir, f'LFIQA_result_{dataset_name}_s.h5')
    with h5py.File(results_file, 'w') as f:
        f.create_dataset('predict_data', data=data_list)
        f.create_dataset('score_label', data=label_list)
    print(f"Results saved in {results_file}")

if __name__ == '__main__':
    # 新增：解析命令行参数，接收权重路径和数据集路径
    parser = argparse.ArgumentParser(description='测试Win5数据集')
    parser.add_argument('weights_path', help='模型权重文件夹路径（如./Win5Nets/）')
    parser.add_argument('dataset_path', help='测试集根目录路径（如./Sci_PVBLiF_Win5.../）')
    args = parser.parse_args()
    
    # 调用测试函数，传入外部路径
    test_model(args.weights_path, args.dataset_path)
