import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpImage
import os
import random
from preprocess import *


def mask2one_hot(num_classes, label, h, w):
   one_hots = []
   for i in range(num_classes):
       tmplate = torch.ones(1, h, w)
       tmplate[label != i] = 0
       one_hots.append(tmplate)

   onehot = torch.cat(one_hots, dim=0)
   return onehot


def get_data_list(dem_data_folder, target_data_folder):
    all_data_path = []
    for f in os.listdir(dem_data_folder):
        dem_path = os.path.join(dem_data_folder, f)
        target_path = os.path.join(target_data_folder, f.split('.')[0] + '.png')
        label = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        if label.sum() > label.shape[0]*label.shape[1]*0.01:
            all_data_path.append((dem_path, target_path))
    return all_data_path * 2


class DTM_Dataset(Dataset):
    def __init__(self, all_data, fine_size, num_classes):
        self.all_data = all_data
        self.padding = 4
        self.fine_size = fine_size   # (512,512) --- H W
        self.num_classes = num_classes

    def __getitem__(self, index):
        dem_path, target_path = self.all_data[index]
        dem_map, proj, geotrans, data_type = readtif(dem_path)
        dem_img = dem_map[self.padding:self.padding*(-1), self.padding:self.padding*(-1)]

        dem = get_constant2onezero(dem_img)

        tpi = get_TPI(dem_map, self.padding)
        slope = get_slope(dem_map, self.padding)
        # curvature = get_curvature(dem_map, self.padding)

        # flowdir = get_flowdir(dem_map, self.padding)
        # flowacc =get_flowacc(dem_map, self.padding, 'D8')
        # DG = get_vectors_from_d8(torch.Tensor(flowdir))
        input_layer = [dem, tpi, slope]
        x_list = []
        for layer in input_layer:
            x_list.append(np.expand_dims(layer, axis=0))
        x_numpy = np.concatenate(x_list, axis=0)        # (C, H, W)
        target_map = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)       # read label file 8bit gray figure
        target_map[target_map > 0] = 1
        if len(target_map.shape) == 2:
            y_numpy = target_map[self.padding:self.padding*(-1), self.padding:self.padding*(-1)]
            y_numpy = np.expand_dims(y_numpy, axis=0)   # (C, H, W)
        random_row = int(random.random()*(dem_img.shape[0] - self.fine_size[0]))
        random_col = int(random.random()*(dem_img.shape[1] - self.fine_size[1]))
        x_numpy = x_numpy[:, random_row:random_row + self.fine_size[0], random_col:random_col + self.fine_size[1]]
        y_numpy = y_numpy[:, random_row:random_row + self.fine_size[0], random_col:random_col + self.fine_size[1]]
        if random.random() < 0.5:
            x_numpy = np.swapaxes(x_numpy, 1, 2)
            y_numpy = np.swapaxes(y_numpy, 1, 2)
        if random.random() < 0.5:
            x_numpy = np.flip(x_numpy, axis=1)
            y_numpy = np.flip(y_numpy, axis=1)
        if random.random() < 0.5:
            x_numpy = np.swapaxes(x_numpy, 1, 2)
            y_numpy = np.swapaxes(y_numpy, 1, 2)
        if random.random() < 0.5:
            x_numpy = np.flip(x_numpy, axis=2)
            y_numpy = np.flip(y_numpy, axis=2)
        if random.random() < 0.5:
            x_numpy = np.swapaxes(x_numpy, 1, 2)
            y_numpy = np.swapaxes(y_numpy, 1, 2)
        if random.random() < 0.5:
            x_numpy = np.flip(x_numpy, axis=1)
            y_numpy = np.flip(y_numpy, axis=1)
        x_samples, y_samples = torch.Tensor(x_numpy.astype(np.float32)), torch.IntTensor(y_numpy.astype(np.uint8))
        labels = mask2one_hot(self.num_classes, y_samples, self.fine_size[0], self.fine_size[1])
        return x_samples, labels

    def __len__(self):
        return len(self.all_data)

