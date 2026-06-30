import os
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import numpy as np
import h5py


class MyTrainSetLoader_Kfold(Dataset):
    def __init__(self, dataset_dir, test_scene_id):
        # 接受数据集目录和要测试的场景id
        super(MyTrainSetLoader_Kfold, self).__init__()
        self.dataset_dir = dataset_dir
        scene_list = ['Bikes', 'dishes', 'Flowers', 'greek', 'museum', 'Palais_du_Luxembourg', 'rosemary', 'Sphynx',
                      'Swans_1', 'Vespa']
        print(scene_list)
        scene_list.pop(test_scene_id[0]) #使用 pop 方法移除 test_scene_id 中指定的两个场景，这样就只保留用于训练的场景。
        scene_list.pop(test_scene_id[1]-1) #使用 pop 方法移除 test_scene_id 中指定的两个场景，这样就只保留用于训练的场景。
        all_patch_path = []
        # 读取所有剩下的场景，构建包含所有补丁路径的列表
        for scene in scene_list:
            # 获取当前场景的路径
            scene_path = os.path.join(dataset_dir, scene)
            # 获取该场景下的所有扭曲场景
            distorted_scene_list = os.listdir(scene_path)  # 读取扭曲场景列表
            for distorted_scene in distorted_scene_list:
                distorted_path_list = os.listdir(os.path.join(scene_path, distorted_scene))
                for distorted_path in distorted_path_list:
                    path = os.path.join(scene, distorted_scene, distorted_path)  # 构建完整路径
                    all_patch_path.append(path)
                    # 最终的path就保存了所有补丁的路径 这里是也按补丁的顺序的。
        self.all_patch_path = all_patch_path
        self.item_num = len(self.all_patch_path)

    def __getitem__(self, index):
        # 用于读取每一个h5
        all_patch_path = self.all_patch_path
        dataset_dir = self.dataset_dir #数据集的名字
        file_name = dataset_dir + '/' + all_patch_path[index]
        with h5py.File(file_name, 'r') as hf:
            data = np.array(hf.get('dis_data'))
            data = data / 255 #归一化
            data = np.transpose(data,[1,2,0]) #假设data是一个3维数组，其形状为(x, y, z)，
            # 那么np.transpose(data, [1, 2, 0])将会得到一个新的数组，其形状为(y, z, x)。
            score_label = np.array(hf.get('score_label'))
        return ToTensor()(data.copy()), ToTensor()(score_label.copy())
    # 返回数据和标签张量。
    def __len__(self):
        return self.item_num
    # 返回样本总数

if __name__ == "__main__":
    # 示例路径和测试场景ID，实际使用时请根据需要修改
    dataset_dir = './Data/PVBLiF_NBU_9_9_32_32_patch/'
    test_scene_id = [0, 1]  # 选择要排除的场景ID
    dataset = MyTrainSetLoader_Kfold(dataset_dir, test_scene_id)

    # 打印 all_patch_path
    print("All Patch Paths:")
    for path in dataset.all_patch_path:
        print(path)

    import torch
    print(torch.cuda.is_available())
    print(torch.__version__)