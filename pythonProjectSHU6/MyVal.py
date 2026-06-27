import torch
from MyDataLoader import *
from alchemical_furnace8 import Network
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from scipy.stats import spearmanr as SRCC


def val(valset_dir, test_scene_id, load_model_path, type = 'val'):

    device = 'cuda:0'
    net = Network().to(device)
    cudnn.benchmark = True
    # model = torch.load(load_model_path, map_location={'cuda:0': device})
    model = torch.load(load_model_path, map_location=device)
    net.load_state_dict(model['state_dict'])
    net.eval()
    label_list = []
    data_list = []
    scene_list = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8']
    for test_scene in test_scene_id:
        image_path = valset_dir + '/' + scene_list[test_scene]
        image_list = os.listdir(image_path)
        for test_image in image_list:
            patch_path = image_path + '/' + test_image
            patch_list = os.listdir(patch_path)
            output_list = 0
            for val_patch in patch_list:
                each_patch_path = patch_path + '/' + val_patch
                with h5py.File(each_patch_path, 'r') as hf:
                    label = np.array(hf.get('score_label'))
                    data = np.array(hf.get('dis_data'))
                    data = data / 255
                    data = np.expand_dims(data, axis=0)
                    data = np.expand_dims(data, axis=0)
                    data = torch.from_numpy(data.copy())
                    data = Variable(data).to(device)
                with torch.no_grad():
                    out_score = net(data)
                output_list += out_score.cpu().numpy().item()
            label_list.append(label.item())
            data_list.append(output_list / len(patch_list))

    loss = torch.nn.MSELoss().to(device)(torch.tensor(data_list), torch.tensor(label_list))
    val_SRCC = SRCC(data_list, label_list).correlation
    #此处计算的SRCC只是简单的取平均 相加除以总数
    print(type + ' SRCC :----    %f' % val_SRCC)
    print(type + ' LOSS :----    %f' % loss)