import torch
import os
from numpy import random
import warnings
import math
warnings.filterwarnings("ignore")
import numpy as np


def readtif(DEM_filepath, is_resample=False, scale=1):
    dataset = gdal.Open(DEM_filepath)
    data = dataset.ReadAsArray()
    data = np.around(data, 3)

    geotrans = list(dataset.GetGeoTransform())
    proj = dataset.GetProjection()
    band_count = dataset.RasterCount
    data_type = dataset.GetRasterBand(1).DataType

    del dataset
    return data, proj, geotrans, data_type


def resampling(source_file, target_file, scale=1.0):
    dataset = gdal.Open(source_file, gdalconst.GA_ReadOnly)
    band_count = dataset.RasterCount
    origin_cols = dataset.RasterXSize  
    origin_rows = dataset.RasterYSize  
    cols = int(origin_cols * scale)
    rows = int(origin_rows * scale)
 
    geotrans = list(dataset.GetGeoTransform())
    print(dataset.GetGeoTransform())
    print(geotrans)
    geotrans[1] = geotrans[1] / scale  
    geotrans[5] = geotrans[5] / scale  
    print(geotrans)
 
    band1 = dataset.GetRasterBand(1)
    data_type = band1.DataType
    target = dataset.GetDriver().Create(target_file, xsize=cols, ysize=rows, bands=band_count,
                                        eType=data_type)
    target.SetProjection(dataset.GetProjection())
    target.SetGeoTransform(geotrans)
    total = band_count + 1
    for index in range(1, total):
        data = dataset.GetRasterBand(index).ReadAsArray(buf_xsize=cols, buf_ysize=rows)
        out_band = target.GetRasterBand(index)
        out_band.SetNoDataValue(dataset.GetRasterBand(index).GetNoDataValue())
        out_band.WriteArray(data)  
        out_band.FlushCache()
        out_band.ComputeBandStats(False)
    del dataset
    del target


def get_TPI(dem_map, padding, kernel_size=7):
    """
    dem_Map: numpy_array, (H+padding*2, W+padding*2)
    padding: int
    :param kernel_size: int
    :return: (H, W)
    """
    kernel = np.zeros((kernel_size, kernel_size))
    num_cell = kernel_size ** 2
    kernel.fill(-1)
    center = (kernel_size - 1) // 2
    kernel[center, center] = num_cell - 1

    tpi = signal.convolve2d(dem_map, kernel, mode='valid')
    tpi = tpi / (num_cell - 1)
    padding = padding - center
    return tpi[padding:padding*(-1), padding:padding*(-1)]

def get_constant2onezero(dem_img):
    """
    dem_img: numpy_array, (H, W) into model fine_size
    :return: (H, W)
    """
    max_val = dem_img.max()
    min_val = dem_img.min()
    dem_img = (dem_img - min_val)/(max_val - min_val)
    dem_img = dem_img * 10

    return dem_img


def get_slope(dem_map, padding):
    slope_i, slope_j = np.gradient(dem_map)
    slope = np.sqrt(slope_i**2 + slope_j**2)
    slope[np.isnan(slope)] = 0
    return slope[padding:padding*(-1), padding:padding*(-1)]

def get_curvature(dem_map, padding):
    slope_i, slope_j = np.gradient(dem_map)
    slope = np.sqrt(slope_i ** 2 + slope_j ** 2)

    slope_i = np.divide(slope_i, slope)
    slope_j = np.divide(slope_j, slope)

    grad_grad_i, _ = np.gradient(slope_i)
    _, grad_grad_j = np.gradient(slope_j)

    curvature = grad_grad_i + grad_grad_j
    curvature[np.isnan(curvature)] = 0

    return curvature[padding:padding*(-1), padding:padding*(-1)]


def get_flowacc(dem_map, padding, algorithm):
    """
    algorithm: str [Dinf, Quinn, D8, Rho8 ...]
    """
    dem_rd = richdem.rdarray(dem_map, no_data=-9999)
    flowacc = np.array(richdem.FlowAccumulation(dem_rd, method=algorithm))
    
    return flowacc[padding:padding*(-1), padding:padding*(-1)]


def get_mask_from_classify(target_img, grade_classify):
    """
    grade_classify: type==list.  must require restricted_max_val 
    class_num = len(list) - 1, and class_id is range(0, 1, 2, ... , len(list)-2)
    """
    mask = np.zeros_like(target_img)
    for i in range(len(grade_classify)-1):
        index_row, index_col = np.where((grade_classify[i] <= target_img) & (target_img < grade_classify[i+1]))
        mask[index_row, index_col] = i
    return mask


if __name__ == '__main__':
    data_Folder_url = main_config.Data_Folder_URL
    label_Folder_url = main_config.Label_Folder_URL
    labels_mark = main_config.ClassLabels_Mark
    if not os.path.exists(data_Folder_url):
        print('data_Folder_url is not exist: {}'.format(data_Folder_url))
        os._exit(0)
    if not os.path.exists(label_Folder_url):
        os.mkdir(label_Folder_url)
        print('Creat label_Folder_url: {}'.format(label_Folder_url))

    get_label_txt(data_Folder_url, label_Folder_url, labels_mark)

