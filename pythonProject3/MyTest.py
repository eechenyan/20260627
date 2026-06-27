import torch
from MyDataLoader import *
import numpy as np
from alchemical_furnace1 import Network
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from scipy.stats import spearmanr as SROCC

def test_model():

    # ### Win5
    # load_all_model_path = './PreTrainedModels/Win5/'
    # valset_dir = './Datasets/PVBLiF_Win5_5x5_32x32/' #测试集的路径
    # dataset_name = 'Win5'
    # scene_list = ['Bikes', 'dishes', 'Flowers', 'greek', 'museum', 'Palais_du_Luxembourg', 'rosemary', 'Sphynx',
    #               'Swans_1', 'Vespa']
    # test_scene_num = 2 #测试场景数量
    # #此处测试终于懂了 就是训练的时候每一此训练的时候会移除特定的两个场景里面的数据集 这部分就用做这次训练的权重网络的测试集
    # distorted_num = 22 #失真数量有22个
    # scene_num = 10 #场景10个

    ## NBU
    load_all_model_path = './Nets/NBUNets/01'
    valset_dir = "/project/pythonProject3/Data/Sci_PVBLiF_NBU_9_9_32_32_patch/"
    
    dataset_name = 'NBU'
    scene_list = ['I01R0', 'I02R0', 'I03R0', 'I04R0', 'I05R0', 'I06R0', 'I07R0',
                  'I08R0', 'I09R0', 'I10R0', 'I11R0', 'I12R0', 'I13R0', 'I14R0']
    test_scene_num = 2
    distorted_num = 15
    scene_num = 14

    ### SHU
    # load_all_model_path = './PreTrainedModels/SHU/'
    # valset_dir = './Datasets/PVBLiF_SHU_5x5_32x32/'
    # dataset_name = 'SHU'
    # scene_list = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8']
    # test_scene_num = 2
    # distorted_num = 30
    # scene_num = 8

    device = 'cuda:4' #这是GPU
    #device = 'cpu'
    net = Network().to(device)
    cudnn.benchmark = True

    all_model = os.listdir(load_all_model_path)
    #label_list = np.zeros([test_scene_num * distorted_num, len(all_model)])
    label_list = []
    data_list = np.zeros([test_scene_num * distorted_num, len(all_model)])
    data_list = []
    val_SRCC_all = [] #用于存储所有模型的斯皮尔曼秩相关系数。
    test_scene_id_list = [] #用于存储测试场景的 ID。test_scene_id_list包含了第几次训练的模型的权重文件,所以打印出来的[0.1]那些数据其实并没有很大的意义
    for a in range(scene_num):
        for b in range(a+1,scene_num):
            test_scene_id_list.append([a,b])
    #这里和train里面的testscenelist一摸一样

    for id, model_name in enumerate(test_scene_id_list): #这里是加载每一个模型并进行测试
        #id 是当前组合在 test_scene_id_list 中的索引（从 0 开始）。model_name 是当前索引对应的场景 ID 组合，形式为 [a, b]。
        load_model_path = load_all_model_path + '/' + 'LF_epoch6.pth.tar'
        print(load_model_path)
        # Load model with weights_only=True
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        model = torch.load(load_model_path, map_location={'cuda:0': device})
        # 移除前缀并加载状态字典
        state_dict = model['state_dict']
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('module.', '')  # 移除 'module.' 前缀
            new_state_dict[new_key] = value
    
        net.load_state_dict(new_state_dict)

        net.eval() #此处设置为评估模式
        index = 0
        test_scene_id = [int(model_name[0]),int(model_name[1])]
        test_scene_id = [3,4]
        for test_scene in test_scene_id: #eg test_scene_id是[0,1]的话那么test_scene会先是0然后1。
            image_path = valset_dir + '/' + scene_list[test_scene]
            image_list = os.listdir(image_path)
            print(image_list)
            for test_image in image_list:
                patch_path = valset_dir + '/' + scene_list[test_scene] + '/' + test_image
                #这里的patch_path和train的patch_path一摸一样的
                patch_list = os.listdir(patch_path)  #列出文件
                output_list = 0
                #output_list = []
                VS_list = []
                for val_patch in patch_list:
                    if int(val_patch[:-3]) < int(len(patch_list) / 2):
                        continue
                    each_patch_path = patch_path + '/' + val_patch
                    with h5py.File(each_patch_path, 'r') as hf:
                        label = np.array(hf.get('score_label'))
                        data = np.array(hf.get('dis_data'))
                        data = data / 255
                        data = np.expand_dims(data, axis=0)
                        data = np.expand_dims(data, axis=0)
                        data = torch.from_numpy(data.copy())
                        data = Variable(data).to(device)
                        #VS = np.array(hf.get('entropy')) #此处是加载获得权重
                    with torch.no_grad():
                        out_score,weight = net(data) #将数据送入模型进行预测，并得到预测得分
                        print("out_score is",out_score)
                    # output_list.append(out_score.cpu().numpy().item()*VS.item()) #预测得分还需要和计算出来的权重进行相乘
                    output_list+=out_score.cpu().numpy().item()
                    #VS_list.append(1.00)
                #label_list[index, id] = label.item()
                label_list.append(label.item())
                #data_list[index, id] = sum(output_list) / sum(VS_list)#每一个patch乘上权重之后还要相加才是图像的总得分
                data_list.append(output_list / len(patch_list))
                index += 1

        # val_SRCC = SROCC(data_list[:,id], label_list[:,id]).correlation
        val_SRCC = SROCC(data_list, label_list).correlation
        #选择所有行的数据，特定列为 id
        val_SRCC_all.append(val_SRCC)
        print(test_scene_id)
        print('SROCC :----    %f' % val_SRCC)
        # 这里的SROCC是使用了加权平均去做的
    print('Average SROCC :----   %f' % np.mean(val_SRCC_all))

    # save in h5 file and test in matlab
    f = h5py.File('./Results/PVBLiF_result_' + dataset_name + '.h5', 'w')
    f.create_dataset('predict_data', data=data_list)
    f.create_dataset('score_label', data=label_list)
    f.close()

if __name__ == '__main__':
    test_model()
