# -*- coding: utf-8 -*-
"""
Fuse two images using pseudo color to encode one image and superimposing on the other.
"""

# %%

import numpy as np
import math
from utils import transformations as tfms
from os import path
import cv2


def array_normalize(input_array):
    max_value = np.max(input_array)
    min_value = np.min(input_array)
    # print('max {}, min {}'.format(max_value, min_value))
    k = 255 / (max_value - min_value)
    min_array = np.ones_like(input_array) * min_value
    normalized = k * (input_array - min_array)
    return normalized

def fuse_images(img_ref, img_folat, alpha=0.4):
    """
    """
    mask = (img_folat > 5).astype(np.float32)
    # print(alpha)
    mask[mask > 0.5] = alpha
    mask_comp = 1.0 - mask

    img_color = cv2.applyColorMap(img_folat, cv2.COLORMAP_JET)
    # print(img_color.shape)

    dst = np.zeros((img_folat.shape[0], img_folat.shape[1], 3), dtype=np.uint8)

    for i in range(3):
        dst[:, :, i] = (img_ref * mask_comp + img_color[:, :, i] * mask).astype(np.uint8)

    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

    return dst

def scale_volume(input_volume, upper_bound=255, lower_bound=0):
    max_value = np.max(input_volume)
    min_value = np.min(input_volume)

    k = (upper_bound - lower_bound) / (max_value - min_value)
    scaled_volume = k * (input_volume - min_value) + lower_bound
    # print('min of scaled {}'.format(np.min(scaled_volume)))
    # print('max of scaled {}'.format(np.max(scaled_volume)))
    return scaled_volume


# Angles in radian version
def decompose_matrix(trans_matrix):
    eus = tfms.euler_from_matrix(trans_matrix[:3, :3], axes='sxyz')
    # trans = trans_matrix[:3, 3]
    params = np.asarray([trans_matrix[0, 3], trans_matrix[1, 3], trans_matrix[2, 3],
                         eus[0], eus[1], eus[2]])
    return params



def construct_matrix(params, initial_transform=None):
    '''
    '''
    mat = tfms.euler_matrix(params[3], params[4], params[5], 'sxyz')
    mat[:3, 3] = np.asarray([params[0], params[1], params[2]])

    if not initial_transform is None:
        mat = mat.dot(initial_transform)

    return mat

# Angles in degree version
def decompose_matrix_degree(trans_matrix):
    eus = tfms.euler_from_matrix(trans_matrix[:3, :3])
    eus = np.asarray(eus, dtype=np.float) / np.pi * 180.0
    params = np.asarray([trans_matrix[0, 3],
                        trans_matrix[1, 3],
                        trans_matrix[2, 3],
                        eus[0], eus[1], eus[2]])
    return params

def construct_matrix_degree(params, initial_transform=None):
    if not params is np.array:
        params = np.asarray(params, dtype=np.float)

    radians = params[3:] / 180.0 * np.pi
    mat = tfms.euler_matrix(radians[0], radians[1], radians[2], 'sxyz')
    mat[:3, 3] = np.asarray([params[0], params[1], params[2]])

    if not initial_transform is None:
        mat = mat.dot(initial_transform)

    return mat

def get_diff_params_as_label(init_mat, target_mat):
    moving_mat = init_mat.dot(np.linalg.inv(target_mat))
    eulers = np.asarray(tfms.euler_from_matrix(moving_mat[:3, :3], axes='sxyz')) / np.pi * 180
    params_rand = np.concatenate((moving_mat[:3, 3], eulers), axis=0)
    return params_rand
# %%

def decompose_matrix_old(trans_matrix):
    # print('trans_matrix\n{}'.format(trans_matrix))
    tX = trans_matrix[0][3]
    tY = trans_matrix[1][3]
    tZ = trans_matrix[2][3]
    # print('tX {}, tY {}, tZ {}'.format(tX, tY, tZ))

    ''' Use online OpenCV codes '''
    ''' radius to degrees '''
    ''' The output angles are degrees! '''
    angleX, angleY, angleZ = rotationMatrixToEulerAngles(trans_matrix[:3, :3])
    angleX = angleX * 180.0 / np.pi
    angleY = angleY * 180.0 / np.pi
    angleZ = angleZ * 180.0 / np.pi

    return np.asarray([tX, tY, tZ, angleX, angleY, angleZ])

def get_array_from_itk_matrix(itk_mat):
    mat = np.reshape(np.asarray(itk_mat), (3, 3))
    return mat

def rotation_matrix(angle, direction='x'):
    rot_mat = np.identity(3)
    sinX = math.sin(angle)
    cosX = math.cos(angle)
    if direction == 'x':
        rot_mat[1][1] = cosX
        rot_mat[1][2] = -sinX
        rot_mat[2][1] = sinX
        rot_mat[2][2] = cosX
    elif direction == 'y':
        rot_mat[0][0] = cosX
        rot_mat[0][2] = sinX
        rot_mat[2][0] = -sinX
        rot_mat[2][2] = cosX
    else:
        rot_mat[0][0] = cosX
        rot_mat[0][1] = -sinX
        rot_mat[1][0] = sinX
        rot_mat[1][1] = cosX

    return rot_mat


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    # assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    # return np.array([x, y, z])
    return x, y, z


def interpolate_transforms(mat_0, mat_1, ratio=0.5):
    ''' Create a new transform by interpolating between two transforms
        with the given ratio.
    '''

    gt_params = decompose_matrix_degree(mat_0)
    bs_params = decompose_matrix_degree(mat_1)

    md_params = gt_params * ratio + (1.0 - ratio) * bs_params

    md_mat = construct_matrix_degree(md_params)

    return md_mat


def load_gt_registration(folder_path):
    fn_reg = 'coreg.txt'
    fn_reg_refined = 'coreg_refined.txt'

    # By default, load the refined registration
    fn_reg_full = path.join(folder_path, fn_reg_refined)

    if not path.isfile(fn_reg_full):
        fn_reg_full = path.join(folder_path, fn_reg)

    # print('loading {}'.format(fn_reg_full))
    gt_reg = np.loadtxt(fn_reg_full)
    return gt_reg

def coord_rigid_transform(point, mat):
    point = np.append(point, [1])
    trans_pt = np.dot(mat, point)
    trans_pt = trans_pt / trans_pt[3]

    return trans_pt[:3]









