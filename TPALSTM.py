#!/usr/bin/python 3.7
#-*-coding:utf-8-*-


import torch
from torch import nn, optim
from tool import RMSE,MAE,MAPE
from tool import Data_normalizer,Weighted_loss
import random
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
from LoadERA5 import WindDataset
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import time


class TPALSTM(nn.Module):

    def __init__(self, input_size, output_horizon, hidden_size, obs_len, n_layers):
        super(TPALSTM, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, \
                            bias=True, batch_first=True)  # output (batch_size, obs_len, hidden_size)
        self.hidden_size = hidden_size
        self.filter_num = 32
        self.filter_size = 1
        self.output_horizon = output_horizon
        self.attention = TemporalPatternAttention(self.filter_size, \
                                                  self.filter_num, obs_len - 1, hidden_size)
        self.linear = nn.Linear(hidden_size, output_horizon)
        self.n_layers = n_layers

    def forward(self, x):
        batch_size, obs_len, num_features = x.size()
        xconcat = self.relu(self.hidden(x))
        H = torch.zeros(batch_size, obs_len - 1, self.hidden_size)
        ht = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        ct = ht.clone()
        for t in range(obs_len):
            xt = xconcat[:, t, :].view(batch_size, 1, -1)
            out, (ht, ct) = self.lstm(xt, (ht, ct))
            htt = ht.permute(1, 0, 2)
            htt = htt[:, -1, :]
            if t != obs_len - 1:
                H[:, t, :] = htt
        H = self.relu(H)

        # reshape hidden states H
        H = H.view(-1, 1, obs_len - 1, self.hidden_size)
        new_ht = self.attention(H, htt)
        ypred = self.linear(new_ht)
        return ypred


class TemporalPatternAttention(nn.Module):

    def __init__(self, filter_size, filter_num, attn_len, attn_size):
        super(TemporalPatternAttention, self).__init__()
        self.filter_size = filter_size
        self.filter_num = filter_num
        self.feat_size = attn_size - self.filter_size + 1
        self.conv = nn.Conv2d(1, filter_num, (attn_len, filter_size))
        self.linear1 = nn.Linear(attn_size, filter_num)
        self.linear2 = nn.Linear(attn_size + self.filter_num, attn_size)
        self.relu = nn.ReLU()

    def forward(self, H, ht):
        _, channels, _, attn_size = H.size()
        new_ht = ht.view(-1, 1, attn_size)
        w = self.linear1(new_ht)  # batch_size, 1, filter_num
        conv_vecs = self.conv(H)

        conv_vecs = conv_vecs.view(-1, self.feat_size, self.filter_num)
        conv_vecs = self.relu(conv_vecs)

        # score function
        w = w.expand(-1, self.feat_size, self.filter_num)
        s = torch.mul(conv_vecs, w).sum(dim=2)
        alpha = torch.sigmoid(s)
        new_alpha = alpha.view(-1, self.feat_size, 1).expand(-1, self.feat_size, self.filter_num)
        v = torch.mul(new_alpha, conv_vecs).sum(dim=1).view(-1, self.filter_num)

        concat = torch.cat([ht, v], dim=1)
        new_ht = self.linear2(concat)
        return new_ht




if __name__ == "__main__":
    seed = 3
    random.seed(seed)
    torch.manual_seed(seed)
    station = 0
    epoch_size, batch_size = 50, 6400
    checkpoint_interval = 1
    Norm_type = 'maxmin'  # 'maxmin' or 'std'
    use_gpu = False
    gpu_id = 0  # 选择要使用的GPU的ID
    Resume = False
    if Resume:
        resume_epoch = 20
    else:
        resume_epoch = 0
    # 迭代次数和检查点保存间隔
    M = 24  # given the M time steps before time t
    N = 24  # predicts the N time steps after time t
    checkpoint_prefix = 'TPALSTM_station{}_'.format(station)
    log_path = "./logs/TPALSTM_station{}_log".format(station)
    # 设置检查点路径和文件名前缀
    checkpoint_path = "./checkpoints/"
    # Get the current date and time
    current_datetime = datetime.now()
    # Format the current date and time to display only hours and minutes
    formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M')

    # 设置GPU
    if use_gpu:
        device = torch.device('cuda:{}'.format(gpu_id))
        torch.cuda.manual_seed(seed)
    else:
        device = torch.device('cpu')

    # 模型定义和训练
    model = TPALSTM(input_size=8+6, output_horizon=24, hidden_size=24, obs_len=24, n_layers=8).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    if Resume:
        # 加载之前保存的模型参数和优化器状态
        checkpoint_name = checkpoint_prefix + str(resume_epoch) + '.pt'
        checkpoint_file = os.path.join(checkpoint_path, checkpoint_name)
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        pass
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=5, verbose=True)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epoch_size)
    criterion = nn.L1Loss()
    weighted_loss = Weighted_loss()
    normalizer = Data_normalizer(station=station)

    trainset = WindDataset(flag='train', station=station, Norm_type=Norm_type, M=M, N=N)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valiset = WindDataset(flag='vali', station=station, Norm_type=Norm_type, M=M, N=N)
    valiloader = DataLoader(valiset, batch_size=batch_size, shuffle=False)

    # 训练循环
    f = open(log_path + formatted_datetime + '.txt', 'a+')  # 打开文件
    f.write('Norm_type:' + Norm_type + '\n')
    f.close()
    for epoch in range(resume_epoch + 1, epoch_size + 1):
        f = open(log_path + formatted_datetime + '.txt', 'a+')  # 打开文件
        n = 0
        train_rmse, train_mae, train_mape = 0.0, 0.0, 0.0
        test_rmse, test_mae, test_mape = 0.0, 0.0, 0.0
        model.train()
        loop = tqdm((trainloader), total=len(trainloader))
        for (hx, fx, y) in loop: # # hx: torch.Size([B, L, C]); hx: torch.Size([B, L-2, C]); y: torch.Size([B, L, C])
            ########################################################
            x = torch.cat([hx, fx], dim=2)
            x = x.to(torch.float32)
            y = y.to(torch.float32)
            x = x.to(device)
            y = y[:, :, -1].to(device)
            y_hat = model(x)
            ########################################################
            loss = criterion(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_num = loss.detach().cpu().numpy()
            loop.set_description(f'Train Epoch: [{epoch}/{epoch_size}] loss: [{loss_num}]')
            train_mae += loss_num * x.shape[0]
            n += x.shape[0]
        train_mae_loss = train_mae / n

        n = 0
        model.eval()
        with torch.no_grad():
            loop = tqdm((valiloader), total=len(valiloader))
            for (hx, fx, y) in loop:  # hx: torch.Size([B, L, C]); hx: torch.Size([B, L-2, C]); y: torch.Size([B, L, C])
                ########################################################
                x = torch.cat([hx, fx], dim=2)
                x = x.to(torch.float32)
                y = y.to(torch.float32)
                x = x.to(device)
                y = y[:, :, -1]
                y_hat = model(x)
                ########################################################
                y_raw, y_hat = y.numpy(), y_hat.detach().cpu().numpy()
                y_raw, y_hat = normalizer.inverse_target(y_raw, y_hat, target='p', Norm_type=Norm_type)
                loss = criterion(torch.from_numpy(y_raw), torch.from_numpy(y_hat))
                loss_num = loss.numpy()
                loop.set_description(f'Test Epoch: [{epoch}/{epoch_size}] loss: [{loss_num}]')
                test_mae += loss_num * x.shape[0]
                rmse_loss = RMSE(y_hat, y_raw)
                mape_loss = MAPE(y_hat, y_raw, normalizer.max_dict['power'])
                test_rmse += rmse_loss * x.shape[0]
                test_mape += mape_loss * x.shape[0]
                n += x.shape[0]

            f.write('Iter: ' + str(epoch) + ' ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\n')
            print('Iter:', epoch, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        test_mae_loss = test_mae / n
        test_rmse_loss = test_rmse / n
        test_mape_loss = test_mape / n
        lr_scheduler.step()
        print("第%d个epoch的学习率：%f" % (epoch, opt.param_groups[0]['lr']))
        if epoch % checkpoint_interval == 0:
            # 保存模型检查点
            checkpoint_name = checkpoint_prefix + str(epoch) + '.pt'
            model_path = os.path.join(checkpoint_path, checkpoint_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'train_loss': train_mae_loss,
                'test_loss': test_mae_loss,
            }, model_path)
            print('Checkpoint saved:', model_path)

        f.write('Train loss: ' + str(train_mae_loss) + '\n')
        f.write('Test mae loss: ' + str(test_mae_loss) + '\n')
        f.write('Test rmse loss: ' + str(test_rmse_loss) + '\n')
        f.write('Test mape loss: ' + str(test_mape_loss) + '\n')
        print('Train loss:', train_mae_loss, ' Test loss:', test_mae_loss)
        print('===' * 20)
        seg_line = '=======================================================================' + '\n'
        f.write(seg_line)
        f.close()





