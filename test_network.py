
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
import numpy as np
import time
import os
from os import path
import random
from utils import data_loading_funcs as load_func
import SimpleITK as sitk
from networks import generators as gens
from numpy.linalg import inv
from datetime import datetime
import argparse

"""load trained model"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
att_gen = gens.AttentionReg()
att_gen = nn.DataParallel(att_gen)
"""*************************MODEL NAME HERE********************************"""
att_gen.load_state_dict(torch.load(
    # 'results/Gen_AttentionReg_xxxx-xxxxxx_load_model.pth'))
    'results/Gen_AttentionReg_0705-193148_load_model.pth'))
att_gen.eval()
att_gen.cuda().to(device)

"""load test data"""
def scale_volume(input_volume, upper_bound=255, lower_bound=0):
    max_value = np.max(input_volume)
    min_value = np.min(input_volume)

    k = (upper_bound - lower_bound) / (max_value - min_value)
    scaled_volume = k * (input_volume - min_value) + lower_bound
    # print('min of scaled {}'.format(np.min(scaled_volume)))
    # print('max of scaled {}'.format(np.max(scaled_volume)))
    return scaled_volume

gt_trans_fn = path.join('sample', 'gt.txt')
gt_mat = np.loadtxt(gt_trans_fn)
base_mat = np.loadtxt('sample/test/init0/initialization_0.txt')

sample4D = np.zeros((2, 32, 96, 96), dtype=np.ubyte)
sample4D[0, :, :, :] = np.load(path.join('sample/test/init0', 'MR_{}.npy'.format(0)))
sample4D[1, :, :, :] = np.load(path.join('sample/test/init0', 'US_{}.npy'.format(0)))
sample4D = scale_volume(sample4D, upper_bound=1, lower_bound=0)
mat_diff = gt_mat.dot(np.linalg.inv(base_mat))
target = load_func.decompose_matrix_degree(mat_diff)
inputs = np.expand_dims(sample4D, axis=0)

"""feeding test data to the network"""
inputs = torch.from_numpy(inputs).float().to(device)
outputs = att_gen(inputs)
outputs = outputs.data.cpu().numpy()
# add_params = np.reshape(outputs, (outputs.shape[1]))

"""evaluation"""
# registration_mat = load_func.construct_matrix_degree(params=add_params, initial_transform=base_mat)
error = np.sum(np.power(outputs-target,2))/6
print(outputs, target)
print('testing MSE error= {}'.format(error))