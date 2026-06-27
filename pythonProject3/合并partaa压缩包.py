import os

# 定义文件路径
file_parts = [
    r'E:\百度网盘下载的东西\Windowed 5 DOF Light Field Image Database\Win5-LID.tar.gz.aa',
    r'E:\百度网盘下载的东西\Windowed 5 DOF Light Field Image Database\Win5-LID.tar.gz.ab',
    r'E:\百度网盘下载的东西\Windowed 5 DOF Light Field Image Database\Win5-LID.tar.gz.ac'
]

# 合并文件
output_file = r'E:\百度网盘下载的东西\Windowed 5 DOF Light Field Image Database\Win5-LID.tar.gz'

with open(output_file, 'wb') as outfile:
    for part in file_parts:
        with open(part, 'rb') as infile:
            outfile.write(infile.read())

# 解压文件
import tarfile

with tarfile.open(output_file, 'r:gz') as tar:
    tar.extractall(path=r'E:\百度网盘下载的东西\Windowed 5 DOF Light Field Image Database\Extracted')

print("解压完成")