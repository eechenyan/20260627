import scipy.io
import matplotlib.pyplot as plt
import numpy as np

# 替换为你的.mat文件路径
mat_file_path = 'D:\\数据集\\LFIQA\\上海大学光场数据库\\distortP1\\P1I1R1_5DLF.mat'

# 使用scipy.io.loadmat读取.mat文件
data = scipy.io.loadmat(mat_file_path)

# 获取'image_data'变量
image_data = data['im2']

# 确定要显示的图像数量
count = 0
max_images = image_data.shape[0] * image_data.shape[1]  # 15*15

# 创建一个图像框
plt.figure(figsize=(15, 15))

# 遍历15x15的图像网格
for i in range(image_data.shape[0]):
    for j in range(image_data.shape[1]):
        # 提取单个图像
        single_image = image_data[i, j, :, :, :]

        # 检查图像是否全为0或全为255
        if not (np.all(single_image == 0) or np.all(single_image == 255)):
            # 计算子图的位置
            pos = plt.subplot(image_data.shape[0], image_data.shape[1], count + 1)
            # 显示图像
            plt.imshow(single_image)
            plt.axis('off')  # 不显示坐标轴
            count += 1
            # 如果已经显示了所有的图像，则退出循环
            if count >= max_images:
                break
    if count >= max_images:
        break

# 调整子图间距
plt.tight_layout()
# 显示图像框
plt.show()