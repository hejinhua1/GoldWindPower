import math
import torch.nn.functional as F
from torch.autograd import Variable
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
import numpy as np
################
# TransformerModel
################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, input_size,output_size,d_model,seq_len):
        super(TransformerModel, self).__init__()
        self.input_fc = nn.Linear(input_size, d_model)
        self.output_fc = nn.Linear(input_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=4 * d_model,
            dropout=0.1,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=8,
            dropout=0.1,
            dim_feedforward=4 * d_model,
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=5)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=5)
        self.fc = nn.Linear(output_size * d_model, output_size)
        self.fc1 = nn.Linear(seq_len * d_model, d_model)
        self.fc2 = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.input_fc(x)
        x = self.pos_emb(x)
        x = self.encoder(x)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        out = self.fc2(x)

        return out

if __name__ == "__main__":
    # x = torch.randn(32, 24, 14)
    # model = TransformerModel(input_size=14, output_size=24, d_model=16, seq_len=24)
    # out = model(x)
    random.seed(2023)
    Resume = False
    if Resume:
        resume_epoch = 20
    else:
        resume_epoch = 0
    # 迭代次数和检查点保存间隔
    gpu_id = 0  # 选择要使用的GPU的ID
    M = 24  # given the M time steps before time t
    N = 24  # predicts the N time steps after time t
    epoch_size, batch_size = 50, 500
    checkpoint_interval = 1
    checkpoint_prefix = 'Transformer_h{}_'.format(N)
    log_path = "E:/HJHCloud/Seafile/startup/GoldWindPower/logs/Transformer_h{}_log".format(N)
    Norm_type = 'maxmin'  # 'maxmin' or 'std'
    # 设置检查点路径和文件名前缀
    checkpoint_path = "E:/HJHCloud/Seafile/startup/GoldWindPower/checkpoints/"
    # Get the current date and time
    current_datetime = datetime.now()
    # Format the current date and time to display only hours and minutes
    formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M')

    # 设置GPU
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(gpu_id))
    else:
        device = torch.device('cpu')

    # 模型定义和训练
    model = TransformerModel(input_size=14, output_size=24, d_model=16, seq_len=24).to(device)
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
    normalizer = Data_normalizer()

    trainset = WindDataset(flag='train', Norm_type=Norm_type, M=M, N=N)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valiset = WindDataset(flag='vali', Norm_type=Norm_type, M=M, N=N)
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
                y = y[:, :, -1].to(device)
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





