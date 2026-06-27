import torch
from MyDataLoader import *
import numpy as np
from alchemical_furnace7 import Network
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from scipy.stats import spearmanr as SROCC
from logistic import verify_performance

# 版本控制手动
# 6.29 将cuda5 修改成了 cuda7

def test_model():

    ### Win5
    # load_all_model_path = '/project/pythonProject3/Netss/Win5Nets/'
    # valset_dir = '/project/pythonProject3/Data/Sci_PVBLiF_Win5_9_9_32_32_patch/' #测试集的路径
    # dataset_name = 'Win5'
    # scene_list = ['Bikes', 'dishes', 'Flowers', 'greek', 'museum', 'Palais_du_Luxembourg', 'rosemary', 'Sphynx',
    #                'Swans_1', 'Vespa']
    # test_scene_num = 2 #测试场景数量
    #此处测试终于懂了 就是训练的时候每一此训练的时候会移除特定的两个场景里面的数据集 这部分就用做这次训练的权重网络的测试集
    # distorted_num = 22 #失真数量有22个
    # scene_num = 10 #场景10个

    ## NBU
    # load_all_model_path = "/project/pythonProject3/Nets/NBUNets"
    # valset_dir = "/project/pythonProject3/Data/Sci_PVBLiF_NBU_9_9_32_32_patch/"
    # dataset_name = 'NBU'
    # scene_list = ['I01R0', 'I02R0', 'I03R0', 'I04R0', 'I05R0', 'I06R0', 'I07R0',
    #               'I08R0', 'I09R0', 'I10R0', 'I11R0', 'I12R0', 'I13R0', 'I14R0']
    # test_scene_num = 2
    # distorted_num = 15
    # scene_num = 14

    ### SHU
    load_all_model_path = '/project/pythonProject3/Netss/SHUNets/'
    #load_all_model_path = '/project/pythonProjectSHU6/logTF0417/'
    
    valset_dir = '/project/pythonProject3/Data/Sci_PVBLiF_SHU_9_9_32_32_patch/'
    dataset_name = 'SHU'
    scene_list = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8']
    test_scene_num = 2
    distorted_num = 30
    scene_num = 8
    
        # 确保结果存储路径存在
    results_dir = './Results/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    device = 'cuda:7' #这是GPU
    net = Network().to(device)
    cudnn.benchmark = True

    all_model = os.listdir(load_all_model_path)
    label_list = np.zeros([test_scene_num * distorted_num, len(all_model)])
    #label_list = []
    data_list = np.zeros([test_scene_num * distorted_num, len(all_model)])
    val_SRCC_all = [] #用于存储所有模型的斯皮尔曼秩相关系数。
    val_SRCCme_all = []
    val_KROCC_all = []
    val_PLCC_all = []
    val_RMSE_all = []
    test_scene_id_list = [] #用于存储测试场景的 ID。test_scene_id_list包含了第几次训练的模型的权重文件,所以打印出来的[0.1]那些数据其实并没有很大的意义
    
    #for a in range(scene_num):
        #for b in range(a+1,scene_num):
            #test_scene_id_list.append([a,b])
    test_scene_id_list.extend([[0,1],[0,2],[0,3],[0,4],[3,6],[3,4]])
    #test_scene_id_list.append([0,1])
    #这里和train里面的testscenelist一摸一样

    for id, model_name in enumerate(test_scene_id_list): #这里是加载每一个模型并进行测试
        #id 是当前组合在 test_scene_id_list 中的索引（从 0 开始）。model_name 是当前索引对应的场景 ID 组合，形式为 [a, b]。
        load_model_path = load_all_model_path + '/' + \
                          str(model_name[0])  + '_' + str(model_name[1]) + '/PVBLiF_epoch.pth.tar'
        # Load model with weights_only=True
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        model = torch.load(load_model_path, map_location={'cuda:0': device}, weights_only=False)

        # Load state_dict with strict=False to ignore missing or unexpected keys
        net.load_state_dict(model['state_dict'], strict=False)
        net.eval() #此处设置为评估模式
        index = 0
        test_scene_id = [int(model_name[0]),int(model_name[1])]
        print('test_scene_id is :', test_scene_id)
        for test_scene in test_scene_id: #eg test_scene_id是[0,1]的话那么test_scene会先是0然后1。
            image_path = valset_dir + '/' + scene_list[test_scene]
            image_list = os.listdir(image_path)
            for test_image in image_list:
                patch_path = valset_dir + '/' + scene_list[test_scene] + '/' + test_image
                #这里的patch_path和train的patch_path一摸一样的
                patch_list = os.listdir(patch_path)  #列出文件
                #output_list = 0
                output_list = []
                VS_list = []
                for val_patch in patch_list:
                    if int(val_patch[:-3]) < int(len(patch_list) / 30):
                    #原来是2 取一半 现在目测5更好
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
                        VS = np.array(hf.get('SCI')) #此处是加载获得权重
                    with torch.no_grad():
                        out_score = net(data) #将数据送入模型进行预测，并得到预测得分
                        #print(out_score)
                    output_list.append(out_score.cpu().numpy().item()*VS.item()) #预测得分还需要和计算出来的权重进行相乘
                    #output_list+=out_score.numpy().cpu.item()
                    VS_list.append(VS.item())
                label_list[index, id] = label.item()
                #label_list.append(label.item())
                data_list[index, id] = sum(output_list) / sum(VS_list)
                #每一个patch乘上权重之后还要相加才是图像的总得分
                #print(data_list[index, id])
                #data_list.append(output_list / len(patch_list))
                index += 1

        val_SRCC = SROCC(data_list[:,id], label_list[:,id]).correlation
        srocc, krocc, plcc, rmse = verify_performance(label_list[:, id], data_list[:, id])

        #选择所有行的数据，特定列为 id
        val_SRCCme_all.append(srocc)
        val_KROCC_all.append(krocc)
        val_PLCC_all.append(plcc)
        val_RMSE_all.append(rmse)
        print(test_scene_id)
        print('SROCC :----    %f' % val_SRCC)
        print(f"Spearmanme's rank correlation coefficient (SROCC): {srocc}")
        print(f"Spearmanme's rank correlation coefficient (KROCC): {krocc}")
        print(f"Pearson linear correlation coefficient (PLCC): {plcc}")
        print(f"Root Mean Squared (RMSE): {rmse}")

        #val_SRCC = SROCC(data_list, label_list).correlation
        #选择所有行的数据，特定列为 id
        val_SRCC_all.append(val_SRCC)
        # 这里的SROCC是使用了加权平均去做的
    print('Average SROCC :----   %f' % np.mean(val_SRCC_all))
    print('Average SROCCme :----   %f' % np.mean(val_SRCCme_all))
    print('Average KROCC :----   %f' % np.mean(val_KROCC_all))
    print('Average PLCC :----   %f' % np.mean(val_PLCC_all))
    print('Average RMSE :----   %f' % np.mean(val_RMSE_all))

    results_file = os.path.join(results_dir, f'LFIQA_result_{dataset_name}_s.h5')
    with h5py.File(results_file, 'w') as f:
        f.create_dataset('predict_data', data=data_list)
        f.create_dataset('score_label', data=label_list)
    print(f"Results saved in {results_file}")

if __name__ == '__main__':
    test_model()
