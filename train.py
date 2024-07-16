import argparse
import datetime
import os
import time

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from utils.dataset_ST import patients_dataset, Patient, dataAug
from model.network_model_ST import CNN_ST
from utils.config import get_config

import csv
from shutil import copyfile
import pandas as pd
import torchio as tio
import glob

default_config = get_config()
train_config = default_config.train

# apply config
dataset =           train_config['dataset']
train_input =       dataset.split('/')[-1].split('_k')[0]
data_type =         dataset.split('/')[-2].split('pad_')[1]
curr_fold =         train_config.curr_fold
batch =             train_config.batch
lr =                train_config.lr
loss_type =         train_config.loss_type
ckpt_path =         train_config.ckpt_path
Optimizer_type =    train_config.Optimizer_type
fold_df =           train_config.fold_excel
stop_num =          train_config.stop_num
max_epoch =         train_config.max_epoch
train_val_ratio =   train_config.train_val_ratio
Augmentation =      train_config.Augmentation

gpu =               torch.device('cuda:{}'.format(train_config.gpu))
torch.cuda.set_device(gpu)
torch.multiprocessing.set_sharing_strategy('file_system')

str_time = time.strftime('%m-%d_%H-%M-%S', time.localtime(time.time()))
print('Training fold: ', curr_fold)
print(fold_df)
model_path = os.path.join('exp/0830_ST_' + train_input + '_T{0}V{1}_{2}_lr{3}_batch{4}_{5}_{6}_10subjects_5folds_T10norm_Aug'.format(
                               int(train_val_ratio*10), 10-int(train_val_ratio*10), loss_type, lr, batch, data_type, Optimizer_type))


if loss_type == 'MSE':
    loss_func = nn.MSELoss()
elif loss_type == 'MAE':
    loss_func = nn.L1Loss()
elif loss_type == 'SmoothL1':
    loss_func = nn.SmoothL1Loss()

print('model save to:', model_path.split('/')[-1])


def train(train_set, valid_set, test_set, fast_m, optimizer, fold, augmentation, name='train'):
    '''
    :param train_set:
    :param valid_set:
    :param fast_m:
    :param optimizer:
    :param name:
    :return:
    '''
    copyfile('config.toml', model_path + '/config.toml')
    copyfile('./model/network_model_ST.py', model_path + '/network_model_ST.py')
    copyfile('./utils/dataset_ST.py', model_path + '/dataset_ST.py')
    min_loss = torch.FloatTensor([float('inf'), ])
    #max_pear = torch.FloatTensor([0, ])

    global stop_num
    ealy_stop = stop_num

    #################### Learning Curve ####################
    LossPath = model_path  + '/Fold' + fold

    with open(LossPath + '/learning_curve.csv', 'a', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['Current Epoch', 'Train {} Loss'.format(loss_type), 'Val {} Loss'.format(loss_type), 'Test {} Loss'.format(loss_type), 
                    'Train Pearson Corr', 'Val Pearson Corr', 'Test Pearson Corr'])
        f.close()
    #################### Learning Curve ####################

    for epoch in range(max_epoch):
        loss_array_train,  loss_array_val, loss_array_test= [], [], []           #add_0901
        pearson_corr_train, pearson_corr_val, pearson_corr_test = [], [], []       #add_0901

########################################################## Train ##########################################################

        dataloader = DataLoader(dataset=train_set, batch_size=batch, shuffle=True, num_workers=2)

        fast_m.train()
        for count, data in enumerate(dataloader): 
            dce_spatial, data_ref, paramters, pos = data # load dce, ktrans, 4d, muscle dce 
            if augmentation:
                dce_spatial = dataAug(dce_spatial)
    		
            dce_spatial = dce_spatial.cuda()
            data_ref = data_ref.cuda()
            paramters = paramters.cuda()

            pre = fast_m(dce_spatial, data_ref) #CNN predict output

            paramters= paramters[:, 0]
            paramters_th = paramters[paramters<=0.05]
            pre_th = pre[paramters<=0.05]
            pre_loss = loss_func(pre_th[:,0], paramters_th)    # pre -> CNN predict results; parameters(obtained from NLLS fitting) -> ground truth
            
            loss = pre_loss
            loss_array_train.append(loss.item())        #add_0901

            x = pre_th[:,0]
            y = paramters_th
            vx = x - torch.mean(x)
            vy = y - torch.mean(y)
            pearson_corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
            pearson_corr_train.append(pearson_corr.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

########################################################## Validation ##########################################################
        with torch.no_grad():
            fast_m.eval()
            valid_load = DataLoader(dataset=valid_set, batch_size=batch, shuffle=False, num_workers=2)
            for count, data in enumerate(valid_load):
                dce_spatial, data_ref, paramters, pos = data   

                dce_spatial = dce_spatial.cuda()
                data_ref = data_ref.cuda()
                paramters = paramters.cuda()

                pre = fast_m(dce_spatial, data_ref)

                paramters= paramters[:, 0]
                paramters_th = paramters[paramters<=0.05]
                pre_th = pre[paramters<=0.05]
                pre_loss = loss_func(pre_th[:,0], paramters_th) 

                v_loss = pre_loss
                loss_array_val.append(v_loss.item())

                x = pre_th[:,0]
                y = paramters_th
                vx = x - torch.mean(x)
                vy = y - torch.mean(y)
                pearson_corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
                pearson_corr_val.append(pearson_corr.item())

########################################################## Test ##########################################################
        with torch.no_grad():
            fast_m.eval()
            test_load = DataLoader(dataset=test_set, batch_size=batch, shuffle=False, num_workers=2)
            for count, data in enumerate(test_load):
                dce_spatial, data_ref, paramters, pos = data   

                dce_spatial = dce_spatial.cuda()
                data_ref = data_ref.cuda()
                paramters = paramters.cuda()

                pre = fast_m(dce_spatial, data_ref)

                paramters= paramters[:, 0]
                paramters_th = paramters[paramters<=0.05]
                pre_th = pre[paramters<=0.05]
                pre_loss = loss_func(pre_th[:,0], paramters_th) 

                t_loss = pre_loss
                loss_array_test.append(t_loss.item())

                x = pre_th[:,0]
                y = paramters_th
                vx = x - torch.mean(x)
                vy = y - torch.mean(y)
                pearson_corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
                pearson_corr_test.append(pearson_corr.item())

        loss_train_avg = np.mean(loss_array_train)
        loss_val_avg = np.mean(loss_array_val)
        loss_test_avg = np.mean(loss_array_test)
        pear_train_avg = np.mean(pearson_corr_train)    
        pear_val_avg = np.mean(pearson_corr_val)       
        pear_test_avg = np.mean(pearson_corr_test)       

        print('epoch: {epoch}, train loss: {train_loss:.6f}, validation loss: {val_loss:.6f}, test loss: {test_loss:.6f}, train pearson: {pear_train_avg:.6f}, val pearson: {pear_val_avg:.6f}, test pearson: {pear_test_avg:.6f}, --stop: {stop}'.format(
                epoch=epoch, train_loss=loss_train_avg, val_loss=loss_val_avg, test_loss=loss_test_avg, 
                pear_train_avg=pear_train_avg, pear_val_avg=pear_val_avg, pear_test_avg=pear_test_avg, stop=ealy_stop))        #add_0824

        #################### Learning Curve ####################
        with open(LossPath + '/learning_curve.csv', 'a', encoding='utf-8', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(['%d' %(epoch), '%.15f' %loss_train_avg, '%.15f' %loss_val_avg, '%.15f' %loss_test_avg, '%.15f' %pear_train_avg, '%.15f' %pear_val_avg, '%.15f' %pear_test_avg])
            f.close()
        #################### Learning Curve ####################

        #################### save weight and early stopping ####################
        if min_loss > loss_val_avg:    
            best_epoch = epoch          
            torch.save(fast_m.state_dict(), os.path.join(model_path, 'Fold'+fold, 'best_{}.tar'.format(name)))
            min_loss = loss_val_avg         
            ealy_stop = stop_num
            if best_epoch > 100 or (best_epoch <= 100 and best_epoch % 10 == 1):
                torch.save(fast_m.state_dict(), os.path.join(model_path, 'Fold'+fold, 'best_{}.tar'.format(epoch)))
        else:                              
            ealy_stop = ealy_stop - 1      
        if ealy_stop and epoch < max_epoch - 1:    
            continue                               
        #################### save weight and early stopping ####################

        return epoch


fold_df = pd.read_excel(fold_df, dtype={'subject':str, 'fold':int})
if not os.path.exists(model_path):
    os.makedirs(os.path.join(model_path))
for f in range(1, len(set(fold_df['fold']))+1):          
    if not os.path.exists(os.path.join(model_path, 'Fold' + str(f))):
        os.makedirs(os.path.join(model_path, 'Fold' + str(f)))


if ckpt_path == 0:
    print('Train the model from scratch!')

for f in range(curr_fold, curr_fold + 1):
    fast_m = CNN_ST().cuda()

    if ckpt_path != 0:
    #ckpt_files = glob.glob(os.path.join('exp', ckpt_path, '*.tar'))
        ckpt_files = os.path.join('exp', ckpt_path)
        if os.path.isfile(ckpt_files):
            fast_m.load_state_dict(torch.load(ckpt_files))
            print('Sucessfully loaded check point from: ', ckpt_files)
        else:
            print('Wrong check point path')
            exit()

    if Optimizer_type == 'Adam':
        optimizer = Adam(fast_m.parameters(), lr=lr)
    elif Optimizer_type == 'SGD':
        optimizer = SGD(fast_m.parameters(), lr=lr)

    train_set_pat, valid_set_pat, test_set_pat = patients_dataset(fold_df, f)

    tune_time = str(datetime.datetime.now())
    t_epoch = train(train_set_pat, valid_set_pat, test_set_pat, fast_m, optimizer, str(f), Augmentation, name='patient')

    print('Fold', f, ', Train start at', tune_time, f'total {t_epoch} epoch')
    print('Fold', f, ', End at', str(datetime.datetime.now()), f'result path:{model_path}')