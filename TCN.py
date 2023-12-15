import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class TCN(nn.Module):
    def __init__(self, input_size, n_layers, k=2, dropout=0.2):  ##nlayers = len(channel_list)
        super().__init__()

        layers = []
        # channel_list = [2 ** i for i in range (n_layers)]
        # channel_list.reverse()
        channel_list = n_layers * [25]

        for i in range(n_layers):
            d = 2 ** i
            if i == 0:
                in_channels = input_size
            else:
                in_channels = channel_list[i - 1]
            out_channels = channel_list[i]

            padding = (k - 1) * d
            layers += [Residual_block(in_channels, out_channels, k, d, padding, dropout)]

        self.net = nn.Sequential(*layers)
        self.linear = nn.Linear(channel_list[-1], 1)

    def init_weights(self):
        # self.linear.weight.data.uniform_(0, 0.01)
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x=None):
        x = x.transpose(1, 2)
        _ = self.net(x)
        out = _.transpose(1, 2)
        out = self.linear(out)
        return out.squeeze()


class Residual_block(nn.Module):
    def __init__(self, input, output, k, d, padding, dropout=0.2):
        super().__init__()
        self.DC_Conv1 = weight_norm(nn.Conv1d(input, output, k, stride=1, padding=padding, dilation=d))
        self.cut1 = cut(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.DC_Conv2 = weight_norm(
            nn.Conv1d(output, output, k, stride=1, padding=padding, dilation=d))  ###这里是output->output
        self.cut2 = cut(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.res = nn.Conv1d(input, output, 1)
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        init_uniform = 0.01
        # self.DC_Conv1.weight.data.uniform_(-init_uniform, init_uniform)
        # self.DC_Conv2.weight.data.uniform_(-init_uniform, init_uniform)
        # self.res.weight.data.uniform_(-init_uniform, init_uniform)

        # self.linear.weight.data.normal_(0, 0.01)
        self.DC_Conv1.weight.data.normal_(0, 0.01)
        self.DC_Conv2.weight.data.normal_(0, 0.01)
        self.res.weight.data.normal_(0, 0.01)

    def forward(self, x):
        res = self.res(x)

        _ = self.DC_Conv1(x)
        _ = self.cut1(_)
        _ = self.relu1(_)
        _ = self.dropout1(_)

        _ = self.DC_Conv2(_)
        _ = self.cut2(_)
        _ = self.relu2(_)
        output = self.dropout2(_)

        return self.relu(output + res)


class cut(nn.Module):
    def __init__(self, size):
        super().__init__()

        self.size = size

    def forward(self, x):
        return x[:, :, :-self.size].contiguous()


class MLP(torch.nn.Module):

    def __init__(self, input_size, output_horizon, hidden_size, obs_len, n_layers=2):
        super(MLP, self).__init__()

        self.linear1 = torch.nn.Linear(input_size * obs_len, hidden_size)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)  # 2个隐层
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(hidden_size, output_horizon)

    def forward(self, x):
        xx = x.view(x.shape[0], -1)
        x = self.linear1(xx)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x


class TCNwithMLP(nn.Module):
    def __init__(self, input_size1, input_size2, output_horizon, obs_len, hidden_size1, hidden_size2, n_layers1,
                 n_layers2=2, k=2, dropout=0.2):
        super().__init__()
        self.TCN1 = TCN(input_size1, n_layers1, k, dropout)
        self.MLP = MLP(input_size2, hidden_size1, hidden_size2, obs_len)
        self.linear = nn.Linear(obs_len + hidden_size1, output_horizon)

    def forward(self, Xw, Xt):
        out1 = self.TCN1(Xt)
        out2 = self.MLP(Xw)
        out = torch.cat([out1, out2], dim=-1)
        out = self.linear(out)
        return out


class TCNwithMLP_change(nn.Module):
    def __init__(self, input_size1, input_size2, output_horizon, obs_len, hidden_size1, hidden_size2, n_layers1,
                 n_layers2=2, k=2, dropout=0.2):
        super().__init__()
        self.TCN1 = TCN(input_size1, n_layers1, k, dropout)
        self.MLP = MLP(input_size2, hidden_size1, hidden_size2, obs_len)
        # self.b = nn.Parameter(torch.Tensor(1))
        # self.c = nn.Parameter(torch.Tensor(1))
        # self.init_weight()

    def init_weight(self):
        self.b.data.uniform_(-1, 1)
        self.c.data.uniform_(-1, 1)

    def forward(self, Xw, Xt):
        out1 = self.TCN1(Xt)
        out2 = self.MLP(Xw)
        out = out1 + out2  ### 固定1效果比使用b，c要好
        return out