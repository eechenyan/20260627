import h5py

# 打开.h5文件
# with h5py.File('Datasets/PVBLiF_NBU_5x5_32x32/I01R0/I01R2/000099.h5', 'r') as file:
with h5py.File(r'D:\PythonProject\pythonProject3\PVBLiF_NBU_9_9_32_32_patch\I14R0\I14R95\000210.h5', 'r') as file:
    # 访问并打印'dis_data'数据集的内容
    dis_data = file['/dis_data'][:]
    print("dis_data:", dis_data)
    print("dis_data的维度",dis_data.shape)

    # 访问并打印'score_label'数据集的内容
    score_label = file['/score_label'][:]
    print("score_label:", score_label)

    # # 访问并打印'VS'数据集的内容
    # score_label = file['/VS'][:]
    # print("VS:", score_label)

