import logging
from networks.CGNet import Context_Guided_Network
from networks.DAM_CGNet import DAM_CGNet
import torch
import torch.nn as nn
import numpy as np
import datetime
from thop import profile

from networks import *
from loss import dice_bce_loss, SegmentationMetric
from dataset import DTM_Dataset, get_data_list
from data import *

torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache

def semantic_segmentation_train(net_name, versions, use_half_training, model, device_id, old_lr, resume, total_epoch,
                                train_data_loader, val_data_loader):
    if use_half_training:
        scaler = torch.cuda.amp.GradScaler()
        autocast = torch.cuda.amp.autocast
        versions = versions + '_half'
    NAME = net_name + '_' + versions  # model_name for saving model weights

    device = None


    if device_id >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(device_id))
        log_handler.info("Training, cuda_is_available. use GPU:{}, {}".format(device, NAME))
    else:
        device = torch.device("cpu")
        log_handler.info("Training, use CPU. model is:".format(NAME))


    model.to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    log_handler.info("Model: {} Total_params: {}".format(net_name, pytorch_total_params))
    

    optimizer = torch.optim.Adam(params=model.parameters(), lr=old_lr, eps=1e-8)
    loss_func = dice_bce_loss()
    EPS = 1e-12

    if resume > 1:
        log_handler.info("Resume training from epoch {}".format(resume))
        model.load_state_dict(torch.load('./weights/' + NAME + '_lastest_Lr.pth'))
   


if __name__ == '__main__':
    in_channel, classNum = 3, 2
    net_name = 'DAM_CGNet'
    versions = '1.0'
    use_half_training = False
    
    device_id = 1
    old_lr = 1e-4  # init learning rate
    resume = 1  # resume > 1: torch.load('./weights/'+ NAME + '_lastest_Lr.pth'))
    total_epoch = 100
    base_batch_size = 6
    

    if net_name == 'CGNet':
        model = Context_Guided_Network(classes=classNum, M=3, N=21)
    elif net_name == 'DAM_CGNet':
        model = DAM_CGNet(classes=classNum, M=3)
    else:
        model = NestedUNet(input_channels=in_channel, output_channels=classNum)
    
    log_handler.info("in_channel : {}, classNum : {}, net_name : {}, versionsm : {}, use_half_training : {}, device_id : {}, old_lr : {}, resume : {}, total_epoch : {}".format(
            in_channel, classNum, net_name, versions, use_half_training, device_id, old_lr, resume, total_epoch))
    
    dem_data_folder = r' DEM storage path '
    target_data_folder = r'  label storage path  '
    all_data_list = get_data_list(dem_data_folder, target_data_folder)
    
    log_handler.info("random_split train_dataSet val_dataSet...........{}".format(len(all_data_list)))
    from sklearn.model_selection import train_test_split
    
    train_data_list, val_data_list = train_test_split(all_data_list, train_size=0.8, test_size=0.2)
    
    train_dataSet = DTM_Dataset(train_data_list, fine_size=[512, 512], num_classes=classNum)
    val_dataSet = DTM_Dataset(val_data_list, fine_size=[512, 512], num_classes=classNum)
    train_data_loader = torch.utils.data.DataLoader(train_dataSet, batch_size=base_batch_size * 4, shuffle=True,
                                                    drop_last=True,
                                                    num_workers=0)
    val_data_loader = torch.utils.data.DataLoader(val_dataSet, batch_size=base_batch_size, shuffle=True, drop_last=True,
                                                  num_workers=0)
    log_handler.info("train_data_loader dataset {}".format(len(train_data_list)))
    log_handler.info("val_data_loader dataset {}".format(len(val_data_list)))
    
    semantic_segmentation_train(net_name, versions, use_half_training, model, device_id, old_lr, resume, total_epoch,
                                train_data_loader, val_data_loader)


