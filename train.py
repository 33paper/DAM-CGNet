import logging
import torch
import torch.nn as nn
import numpy as np
import datetime
from thop import profile

from networks import *
from loss import dice_bce_loss, SegmentationMetric
from dataset import DTM_Dataset, get_data_list
from data import *
from sklearn.model_selection import KFold

torch.backends.cudnn.benchmark = True  
torch.cuda.empty_cache  

log_handler = logging.getLogger('test_logger')   
log_handler.setLevel(logging.INFO)               
test_log = logging.FileHandler(r"./logs/{}.txt".format(str(datetime.datetime.now()).replace(':', '_')), 'a', encoding='utf-8')
test_log.setLevel(logging.INFO)                 
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)                        
formatter = logging.Formatter('%(asctime)s %(levelname)s - %(message)s')   
test_log.setFormatter(formatter)
sh.setFormatter(formatter)
log_handler.addHandler(test_log)             
log_handler.addHandler(sh)                   


def semantic_segmentation_train(net_name, versions, use_half_training, model, device_id, old_lr, resume, total_epoch,
                                train_data_loader, val_data_loader):
    if use_half_training:
        scaler = torch.cuda.amp.GradScaler()     
        autocast = torch.cuda.amp.autocast
        versions = versions + '_half'
    NAME = net_name + '_' + versions  # model_name for saving model weights


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
    try:
        all_train_loss, all_val_loss, all_train_FWIoU, all_val_FWIoU = [], [], [], []
        epoch = 1
        no_optim, no_val_optim, train_epoch_best_loss, val_epoch_best_loss, train_epoch_best_FWIoU, val_epoch_best_FWIoU = 0, 0, 9999, 9999, 0, 0
        for epoch in range(0 + resume, total_epoch + 1):
            dt_size = len(train_data_loader.dataset)
            Iterations = (dt_size - 1) // train_data_loader.batch_size + 1  # drop_last=True, do not add 1
            print_frequency = Iterations // 10  
            gap_frequency = 500000
            model.train()
            train_epoch_loss, train_epoch_FWIoU, train_evalus_res = 0, 0, None
            print_frequency_count = 0
          
            for i, (img, mask) in enumerate(train_data_loader):
                b_x = img.to(device)
                b_y = mask.to(device)
                if not use_half_training:
                    optimizer.zero_grad()
                    output = model(b_x)
                    loss = loss_func(output, b_y, use_half_training)
                    loss.backward()
                    optimizer.step()

                if use_half_training:
                    optimizer.zero_grad()
                    with autocast():
                        output = model(b_x)
                        loss = loss_func(output, b_y, use_half_training)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                if i % print_frequency == 0 and i > 0 and i // gap_frequency < 10:
                    log_handler.info('\ttrain_epoch = %3d | iters = %3d | loss = %3.8f | %s' % (epoch, i, loss.item(), str(datetime.datetime.now())))
                    metric = SegmentationMetric(numClass=output.shape[1], ignore_labels=None)
                    train_evalus_res = metric.evalus(torch.argmax(output, dim=1).cpu(), torch.argmax(b_y, dim=1).cpu())
                    train_iters_FWIoU = float(train_evalus_res.split(':')[-1])
                    all_train_loss.append(np.around(loss.item(), 5))
                    all_train_FWIoU.append(np.around(train_iters_FWIoU, 5))
                    print_frequency_count += 1
                    train_epoch_FWIoU += train_iters_FWIoU
                train_epoch_loss += loss.item() * b_x.size(0)
            train_epoch_loss /= dt_size
            train_epoch_FWIoU /= print_frequency_count
            dt_size = len(val_data_loader.dataset)
            Iterations = (dt_size - 1) // val_data_loader.batch_size + 1  # drop_last=True, do not add 1
            model.eval()      
            torch.no_grad()   

            val_epoch_loss, val_epoch_FWIoU, val_evalus_res = 0, 0, None
            print_frequency_count = 0
            for i, (img, mask) in enumerate(val_data_loader):
                b_x = img.to(device)
                b_y = mask.to(device)
                if not use_half_training:
                    output = model(b_x)
                    loss = loss_func(output, b_y, use_half_training)

                if i % print_frequency == 0 and i > 0 and i // gap_frequency < 10:
                    log_handler.info('\tval_epoch = %3d | iters = %3d | loss = %3.8f | %s' % (epoch, i, loss.item(), str(datetime.datetime.now())))
                    metric = SegmentationMetric(numClass=output.shape[1], ignore_labels=None)
                    val_evalus_res = metric.evalus(torch.argmax(output, dim=1).cpu(), torch.argmax(b_y, dim=1).cpu())
                    val_iters_FWIoU = float(val_evalus_res.split(':')[-1])
                    all_val_loss.append(np.around(loss.item(), 5))
                    all_val_FWIoU.append(np.around(val_iters_FWIoU, 5))
                    print_frequency_count += 1
                    val_epoch_FWIoU += val_iters_FWIoU
                val_epoch_loss += loss.item() * b_x.size(0)
            val_epoch_loss /= dt_size
            val_epoch_FWIoU /= print_frequency_count
            torch.save(model.state_dict(), "./weights/{}_epoch_{}.pth".format(NAME, epoch)) 
            if train_epoch_loss >= train_epoch_best_loss + 0.00:
                no_optim += 1
            else:
                no_optim = 0
                train_epoch_best_loss = train_epoch_loss
                torch.save(model.state_dict(), './weights/' + NAME + '_lastest_Lr.pth')
            if no_optim > 6:
                log_handler.info('The learningRate has been optimised 6 times. Training_Early_Stop ...')
                break
            if no_optim >= 3:
                if old_lr < 5e-7:
                    log_handler.info("{}_EPOCH_{}_smallest_LR_Early_Stop ...".format(str(datetime.datetime.now()), epoch))
                    break
                model.load_state_dict(torch.load('./weights/' + NAME + '_lastest_Lr.pth'))
                new_lr = old_lr / 5 
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                log_handler.info(
                    "{}_Update_Learning_Late_{}_To_{}".format(str(datetime.datetime.now()), old_lr, new_lr))
                old_lr = new_lr
            if val_epoch_FWIoU <= val_epoch_best_FWIoU - 0.00:
                no_val_optim += 1
                if no_val_optim >= 7:
                    log_handler.info("{}_EPOCH_{}_best_FWIoU_Early_Stop ...".format(str(datetime.datetime.now()), epoch))
                    break
            else:
                no_val_optim = 0
                val_epoch_best_FWIoU = val_epoch_FWIoU
                torch.save(model.state_dict(), './weights/' + NAME + '_val_best_FWIoU.pth')
            # end of one epoch
        # end of all epoch
    finally:  # final save checkpoint
        torch.save(model.state_dict(), "./weights/{}_Interrupt_epoch_{}.pth".format(NAME, epoch))


if __name__ == '__main__':
    in_channel, classNum = 3, 2
    net_name = 'DAMCGNet3channel'
    versions = '1.0'
    use_half_training = False
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    device_id = 1
    old_lr = 1e-4  # init learning rate
    resume = 1  # resume > 1: torch.load('./weights/'+ NAME + '_lastest_Lr.pth'))
    total_epoch = 100
    base_batch_size = 6
    log_handler.info("in_channel : {}, classNum : {}, net_name : {}, versionsm : {}, use_half_training : {}, device_id : {}, old_lr : {}, resume : {}, total_epoch : {}".format(
            in_channel, classNum, net_name, versions, use_half_training, device_id, old_lr, resume, total_epoch))
    
    dem_data_folder = r''
    target_data_folder = r''
    all_data_list = get_data_list(dem_data_folder, target_data_folder)
    
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(all_data_list)):
        log_handler.info(f"Fold {fold+1}/{k_folds}")
        
        train_data_list = [all_data_list[i] for i in train_idx]
        val_data_list = [all_data_list[i] for i in val_idx]

        train_dataSet = DTM_Dataset(train_data_list, fine_size=[512, 512], num_classes=classNum)
        val_dataSet = DTM_Dataset(val_data_list, fine_size=[512, 512], num_classes=classNum)

        train_data_loader = torch.utils.data.DataLoader(train_dataSet, batch_size=base_batch_size * 4, shuffle=True,
                                                        drop_last=True, num_workers=0)
        val_data_loader = torch.utils.data.DataLoader(val_dataSet, batch_size=base_batch_size, shuffle=False, drop_last=False,
                                                      num_workers=0)

        log_handler.info(f"Fold {fold+1} train dataset size: {len(train_data_list)}")
        log_handler.info(f"Fold {fold+1} val dataset size: {len(val_data_list)}")

        
        if net_name == 'CGNet': 
            model = Context_Guided_Network(classes=classNum, M=3, N=21) 
        else: 
            model = DAM_CGNet(classes=classNum, M=3, N=21)

        semantic_segmentation_train(net_name, versions, use_half_training, model, device_id, old_lr, resume, total_epoch, 
                                    train_data_loader, val_data_loader)

    log_handler.info("K-Fold cross validation finished.")

 
