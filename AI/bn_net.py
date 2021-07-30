import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.distributions as dis


class MinusOneModule(nn.Module):
    def __init__(self):
        super(MinusOneModule, self).__init__()

    def forward(self, x):
        return x - 1


class BatchNorm1d_LastDim(nn.Module):
    def __init__(self, **kwargs):
        super(BatchNorm1d_LastDim, self).__init__()
        self.bn = nn.BatchNorm1d(**kwargs)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.transpose(1, 2)
        x = self.bn(x)
        if len(x.shape) == 3:
            x = x.transpose(1, 2)
        return x


class MahjongNet(nn.Module):
    def __init__(self, n_channels, batch_norm_tau=0):
        super(MahjongNet, self).__init__()

        self.n_channels = n_channels

        mps_cnn_list = nn.ModuleList()
        mps_cnn_list.append(nn.Conv1d(n_channels, 64, 3, 1, 1))

        if batch_norm_tau:
            mps_cnn_list.append(nn.BatchNorm1d(64, momentum=1 / batch_norm_tau))

        mps_cnn_list.append(nn.ReLU())
        mps_cnn_list.append(nn.Conv1d(64, 64, 3, 1, 1))

        if batch_norm_tau:
            mps_cnn_list.append(nn.BatchNorm1d(64, momentum=1 / batch_norm_tau))

        mps_cnn_list.append(nn.ReLU())
        mps_cnn_list.append(nn.Conv1d(64, 64, 3, 1, 1))

        if batch_norm_tau:
            mps_cnn_list.append(nn.BatchNorm1d(64, momentum=1 / batch_norm_tau))

        mps_cnn_list.append(nn.ReLU())
        mps_cnn_list.append(nn.Flatten())

        self.mps_cnn = nn.Sequential(*mps_cnn_list)
        # 576

        if batch_norm_tau:
            self.wind_mlp = nn.Sequential(nn.Linear(n_channels, 128),
                                          nn.BatchNorm1d(128, momentum=1 / batch_norm_tau),
                                          nn.ReLU(),
                                          nn.Linear(128, 128),
                                          nn.BatchNorm1d(128, momentum=1 / batch_norm_tau),
                                          nn.ReLU(),
                                          nn.Linear(128, 128),
                                          nn.BatchNorm1d(128, momentum=1 / batch_norm_tau),
                                          nn.ReLU(),
                                          )

            self.z_mlp = nn.Sequential(nn.Linear(n_channels, 128),
                                       nn.BatchNorm1d(128, momentum=1 / batch_norm_tau),
                                       nn.ReLU(),
                                       nn.Linear(128, 128),
                                       nn.BatchNorm1d(128, momentum=1 / batch_norm_tau),
                                       nn.ReLU(),
                                       nn.Linear(128, 128),
                                       nn.BatchNorm1d(128, momentum=1 / batch_norm_tau),
                                       nn.ReLU(),
                                       )
        else:
            self.wind_mlp = nn.Sequential(nn.Linear(n_channels, 128),
                                          nn.ReLU(),
                                          nn.Linear(128, 128),
                                          nn.ReLU(),
                                          nn.Linear(128, 128),
                                          nn.ReLU(),
                                          )

            self.z_mlp = nn.Sequential(nn.Linear(n_channels, 128),
                                       nn.ReLU(),
                                       nn.Linear(128, 128),
                                       nn.ReLU(),
                                       nn.Linear(128, 128),
                                       nn.ReLU(),
                                       )

        self.phi_size = 576 + 128 + 128

    def forward(self, x):
        # Shape of x: [batch_size x n_channels x 34]

        man = x[:, :, 0:9]
        pin = x[:, :, 9:18]
        sou = x[:, :, 18:27]

        phi_mps = self.mps_cnn(man) + self.mps_cnn(pin) + self.mps_cnn(sou)
        phi_wind = self.wind_mlp(torch.transpose(x[:, :, 27:31], 1, 2).reshape([-1, self.n_channels])).view(
            [-1, 4, 128]).sum(dim=1)
        phi_z = self.z_mlp(torch.transpose(x[:, :, 31:34], 1, 2).reshape([-1, self.n_channels])).view(
            [-1, 3, 128]).sum(dim=1)

        phi = torch.cat([phi_mps, phi_wind, phi_z], dim=-1)

        return phi


def make_cnn(resolution, n_channels, batch_norm_tau=0):

    if resolution == "34":
        # -------- for Mahjong ---------
        mahjong_net = MahjongNet(n_channels, batch_norm_tau=batch_norm_tau)
        phi_size = mahjong_net.phi_size
        return mahjong_net, phi_size

    else:
        raise NotImplementedError


class DiscreteActionQNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=None, dueling=False, act_fn=nn.ReLU,
                 output_distribution='DiracDelta', batch_norm_tau=0):
        super(DiscreteActionQNetwork, self).__init__()

        if hidden_layers is None:
            hidden_layers = [256, 256]
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.dueling = dueling

        self.output_distribution = output_distribution
        self.network_modules = nn.ModuleList()

        last_layer_size = input_size
        for layer_size in hidden_layers:
            self.network_modules.append(nn.Linear(last_layer_size, layer_size))
            if batch_norm_tau:
                self.network_modules.append(nn.BatchNorm1d(layer_size, momentum=1 / batch_norm_tau))
            self.network_modules.append(act_fn())
            last_layer_size = layer_size

        if self.output_distribution == 'DiracDelta':
            if not dueling:
                self.network_modules.append(nn.Linear(last_layer_size, output_size))
            else:
                self.value_layer = nn.Linear(last_layer_size, 1)
                self.advantage_layer = nn.Linear(last_layer_size, output_size)

        elif self.output_distribution == 'Gaussian':
            if not dueling:
                self.mu_layer = nn.Linear(last_layer_size, output_size)
                # self.logsig_layer = nn.Linear(last_layer_size, output_size)
            else:
                self.value_layer = nn.Linear(last_layer_size, 1)
                self.advantage_layer = nn.Linear(last_layer_size, output_size)
                # self.logsig_layer = nn.Linear(last_layer_size, output_size)

        self.main_network = nn.Sequential(*self.network_modules)

    def forward(self, x):

        if self.output_distribution == 'DiracDelta':

            if not self.dueling:
                q = self.main_network(x)
            else:
                h = self.main_network(x)
                v = self.value_layer(h).repeat_interleave(self.output_size, dim=-1)
                q0 = self.advantage_layer(h)
                a = q0 - torch.mean(q0, dim=-1, keepdim=True).repeat_interleave(self.output_size, dim=-1)
                q = v + a

            return q

        elif self.output_distribution == 'Gaussian':
            if not self.dueling:
                h = self.main_network(x)
                mu_q = self.mu_layer(h)
                logsig = - 3 * torch.ones_like(mu_q)

            else:
                h = self.main_network(x)
                v = self.value_layer(h).repeat_interleave(self.output_size, dim=-1)
                q0 = self.advantage_layer(h)
                a = q0 - torch.mean(q0, dim=-1, keepdim=True).repeat_interleave(self.output_size, dim=-1)
                mu_q = v + a
                logsig = - 3 * torch.ones_like(mu_q)

            dist = dis.normal.Normal(mu_q, torch.exp(logsig))

            return dist.rsample()

    def get_log_prob(self, x, q_target):

        if self.output_distribution == 'DiracDelta':
            return - 0.5 * (self.forward(x) - q_target).pow(2)

        elif self.output_distribution == 'Gaussian':
            if not self.dueling:
                h = self.main_network(x)
                mu_q = self.mu_layer(h)
                logsig = - 3 * torch.ones_like(mu_q)

            else:
                h = self.main_network(x)
                v = self.value_layer(h).repeat_interleave(self.output_size, dim=-1)
                q0 = self.advantage_layer(h)
                a = q0 - torch.mean(q0, dim=-1, keepdim=True).repeat_interleave(self.output_size, dim=-1)
                mu_q = v + a
                logsig = - 3 * torch.ones_like(mu_q)

            dist = dis.normal.Normal(mu_q, torch.exp(logsig))

            return dist.log_prob(q_target)


