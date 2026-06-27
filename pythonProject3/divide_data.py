#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np
import h5py
import cv2  # OpenCV for image processing
from scipy.stats import variation  # For variance calculation
import scipy.io  # To load .mat files
from tqdm import tqdm  # For progress bar

# Set the dataset path here
dataset_path = "/project/LFIQAdataset/NBU-LF1.0/dis_img//"
savepath = "/project/pythonProject3/Data/PVBLiF_NBU_9_9_32_32_patch//"

# Load the data
NBU_all_info = scipy.io.loadmat('/project/LFIQAdataset/NBU-LF1.0/NBU_all_info.mat')
NBU_all_mos = scipy.io.loadmat('/project/LFIQAdataset/NBU-LF1.0/NBU_all_mos.mat')

Distorted_sceneNum = 210  # Number of distorted scenes
angRes = 9  # Angular resolution
patchsize = 32  # Patch size

# Create a progress bar
progress_bar = tqdm(total=Distorted_sceneNum, desc='Processing Scenes')

for iScene in range(Distorted_sceneNum):
    idx_s = 0
    idx = 1
    h5_savedir = os.path.join(savepath, 
                              str(NBU_all_info['NBU_all_info'][0][iScene][0][0].item()),
                              str(NBU_all_info['NBU_all_info'][0][iScene][1][0].item()))

    # Check for valid scene information
    if not NBU_all_info['NBU_all_info'][0][iScene][0][0] or not NBU_all_info['NBU_all_info'][0][iScene][1][0]:
        raise ValueError(f'Invalid scene information at index {iScene}')

    # Replace illegal characters
    h5_savedir = h5_savedir.replace('*', '_')
    os.makedirs(h5_savedir, exist_ok=True)  # Create directory if it doesn't exist

    dataPath = os.path.join(dataset_path, str(NBU_all_info['NBU_all_info'][0][iScene][5][0]))
    if NBU_all_info['NBU_all_info'][0][iScene][2][0] == 'Real':
        LF = np.zeros((9, 9, 434, 625, 3), dtype=np.uint8)
    else:
        LF = np.zeros((9, 9, 512, 512, 3), dtype=np.uint8)

    # Read all images in the scene and store them in the LF array
    for x in range(9):
        for y in range(9):
            dis_single_image = cv2.imread(os.path.join(dataPath, f'00{x}_00{y}.png'))
            LF[y, x, :, :, :] = dis_single_image

    tem_size = LF.shape
    if tem_size[2] == 512:
        total_patch_number = 256
    else:
        LF = LF[:, :, 2:434, 2:624, :]
        hnumber = 433 // patchsize
        wnumber = 623 // patchsize
        hstart = (433 - patchsize * hnumber) // 2
        wstart = (623 - patchsize * wnumber) // 2
        LF = LF[:, :, hstart:hstart + hnumber * patchsize, wstart:wstart + wnumber * patchsize, :]
        total_patch_number = 247

    LF = LF.astype(np.uint8)
    U, V, H, W, A = LF.shape

    # Calculate the saliency score (currently commented out in MATLAB code)
    label = np.array(NBU_all_mos['NBU_all_mos'][0][iScene], dtype=np.float32)

    all_VS_list = []
    var_list = []
    dis_data_mirco = np.zeros((total_patch_number, U * patchsize, V * patchsize), dtype=np.float32)

    for h in range(0, H, patchsize):
        for w in range(0, W, patchsize):
            idx_s += 1
            PVBS_var = []
            for u in range(U):
                for v in range(V):
                    temp_dis = LF[u, v, h:h + patchsize, w:w + patchsize, :]
                    temp_dis_ycbcr = cv2.cvtColor(temp_dis, cv2.COLOR_RGB2YCrCb)
                    temp_dis_y = temp_dis_ycbcr[:, :, 0]
                    dis_data_mirco[idx_s - 1, u:angRes:U * patchsize, v:angRes:V * patchsize] = temp_dis_y
                    PVBS_var.append(variation(temp_dis_y.flatten()))

            var_list.append(np.mean(PVBS_var))

    var_list = np.array(var_list)
    index = np.argsort(var_list)

    for i in range(total_patch_number):
        save_dis_data_mirco = dis_data_mirco[index[i], :, :]
        save_dis_data_mirco_3D = np.zeros((patchsize, patchsize, angRes * angRes), dtype=np.float32)

        for x in range(angRes):
            for y in range(angRes):
                save_dis_data_mirco_3D[:, :, (x * angRes) + y] = save_dis_data_mirco[x:angRes:angRes * patchsize,
                                                                 y:angRes:angRes * patchsize]

        SavePath_H5_name = os.path.join(h5_savedir, f'{idx:06d}.h5')

        with h5py.File(SavePath_H5_name, 'w') as h5file:
            h5file.create_dataset('/dis_data', data=save_dis_data_mirco_3D, dtype='float32')
            h5file.create_dataset('/score_label', data=label, dtype='float32')

        idx += 1

    progress_bar.update(1)

# Close progress bar
progress_bar.close()