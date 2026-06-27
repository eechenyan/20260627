from PIL import Image

# 读取真实图像
real_img = Image.open(r'E:\百度网盘下载的东西\Windowed 5 DOF Light Field Image Database\Extracted\Win5-LID\Distorted\Real/EPICNN_Bikes.bmp')

# 读取合成图像
# synthetic_img = Image.open('./Distorted/Synthetic/EPICNN_dishes.bmp')

# 显示图像（可选）
real_img.show()
# synthetic_img.show()