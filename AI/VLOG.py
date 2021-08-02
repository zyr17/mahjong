import warnings

import gym
import torch
import torch.nn as nn
import numpy as np
import time
from torch.nn import BatchNorm1d
import torch.nn.functional as F
import torch.distributions as dis
from gym.spaces import Box, Discrete
from bn_net import DiscreteActionQNetwork, DiscreteActionPolicyNetwork, MinusOneModule

torch.set_default_dtype(torch.float32)

INFINITY = 1e9


# class BatchNorm1d_LastDim(nn.Module):
#     def __init__(self, **kwargs):
#         super(BatchNorm1d_LastDim, self).__init__()
#         self.bn = nn.BatchNorm1d(**kwargs)
#
#     def forward(self, x):
#         if len(x.shape) == 3:
#             x = x.transpose(1, 2)
#         x = self.bn(x)
#         if len(x.shape) == 3:
#             x = x.transpose(1, 2)
#         return x

def augment_mahjong_data(x):
    assert x.shape[-1] == 34

    agumented_ind = - np.ones([34], dtype=np.int64)

    mps_permu = np.random.permutation(3)

    agumented_ind[0:9] = mps_permu[0] * 9 + np.arange(9)
    agumented_ind[9:18] = mps_permu[1] * 9 + np.arange(9)
    agumented_ind[18:27] = mps_permu[2] * 9 + np.arange(9)

    agumented_ind[27:31] = np.random.permutation(4) + 27
    agumented_ind[31:34] = np.random.permutation(3) + 31

    return x[:, :, agumented_ind]


class MahjongEqualNet(nn.Module):
    def __init__(self, n_channels, batch_norm_tau=0):
        super(MahjongEqualNet, self).__init__()

        self.n_channels = n_channels

        mps_cnn_list = nn.ModuleList()
        mps_cnn_list.append(nn.Conv2d(n_channels, 64, (3, 1), (1, 1), (1, 0)))

        if batch_norm_tau:
            mps_cnn_list.append(nn.BatchNorm2d(64, momentum=1 / batch_norm_tau))

        mps_cnn_list.append(nn.ReLU())
        mps_cnn_list.append(nn.Conv2d(64, 64, (3, 1), (1, 1), (1, 0)))

        if batch_norm_tau:
            mps_cnn_list.append(nn.BatchNorm2d(64, momentum=1 / batch_norm_tau))

        mps_cnn_list.append(nn.ReLU())
        mps_cnn_list.append(nn.Conv2d(64, 64, (3, 1), (1, 1), (1, 0)))

        if batch_norm_tau:
            mps_cnn_list.append(nn.BatchNorm2d(64, momentum=1 / batch_norm_tau))

        mps_cnn_list.append(nn.ReLU())

        self.mps_cnn = nn.Sequential(*mps_cnn_list)
        # 576
        self.flatten = nn.Flatten()

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

        mps = x[:, :, 0:27].reshape([x.size()[0], x.size()[1], 9, 3])

        phi_mps = self.flatten(self.mps_cnn(mps).sum(dim=-1))
        phi_wind = self.wind_mlp(torch.transpose(x[:, :, 27:31], 1, 2).reshape([-1, self.n_channels])).view(
            [-1, 4, 128]).sum(dim=1)
        phi_z = self.z_mlp(torch.transpose(x[:, :, 31:34], 1, 2).reshape([-1, self.n_channels])).view(
            [-1, 3, 128]).sum(dim=1)

        phi = torch.cat([phi_mps, phi_wind, phi_z], dim=-1)

        return phi


class MahjongNet(nn.Module):
    def __init__(self, n_channels, batch_norm_tau=0):
        super(MahjongNet, self).__init__()

        self.n_channels = n_channels

        cnn_list = nn.ModuleList()
        cnn_list.append(nn.Conv1d(n_channels, 64, 3, 1, 1))

        if batch_norm_tau:
            cnn_list.append(nn.BatchNorm1d(64, momentum=1 / batch_norm_tau))

        cnn_list.append(nn.ReLU())
        cnn_list.append(nn.Conv1d(64, 64, 3, 1, 1))

        if batch_norm_tau:
            cnn_list.append(nn.BatchNorm1d(64, momentum=1 / batch_norm_tau))

        cnn_list.append(nn.ReLU())
        cnn_list.append(nn.Conv1d(64, 32, 3, 1, 1))

        if batch_norm_tau:
            cnn_list.append(nn.BatchNorm1d(32, momentum=1 / batch_norm_tau))

        cnn_list.append(nn.ReLU())
        cnn_list.append(nn.Flatten())

        self.cnn = nn.Sequential(*cnn_list)
        # 1088

        self.phi_size = 1088

    def forward(self, x):
        # Shape of x: [batch_size x n_channels x 34]

        phi = self.cnn(x)

        return phi


class DDQN(nn.Module):

    def __init__(self, observation_space, full_observation_space, action_space, is_main_network=True, **kwargs):

        super(DDQN, self).__init__()

        if isinstance(action_space, Box):
            self.action_size = action_space.shape[0]
            self.algorithm = kwargs["algorithm"] if ("algorithm" in kwargs) else 'sac'
        elif isinstance(action_space, Discrete):
            self.action_size = action_space.n
            self.algorithm = kwargs["algorithm"] if ("algorithm" in kwargs) else 'ddqn'
        else:
            raise NotImplementedError

        if isinstance(observation_space, Box) and isinstance(full_observation_space, Box):
            self.input_forward_size = [*observation_space.shape]
            self.input_oracle_size = [*full_observation_space.shape]
        else:
            raise NotImplementedError

        self.tabular_like_reg = kwargs["tabular_like_reg"] if ("tabular_like_reg" in kwargs) else 0

        self.hidden_layer_width = kwargs["hidden_layer_width"] if ("hidden_layer_width" in kwargs) else 256
        self.half_hidden_layer_depth = kwargs["half_hidden_layer_depth"] if ("half_hidden_layer_depth" in kwargs) else 2
        self.act_fn = kwargs["act_fn"] if ("act_fn" in kwargs) else 'relu'

        self.z_stochastic_size = kwargs["z_stochastic_size"] if ("z_stochastic_size" in kwargs) else 128  # not used currently
        self.z_deterministic_size = kwargs["z_deterministic_size"] if ("z_deterministic_size" in kwargs) else 0

        self.beta = kwargs["beta"] if ("beta" in kwargs) else 1.0

        self.kld_target = kwargs["kld_target"] if ("kld_target" in kwargs) else -1
        self.alpha = kwargs["alpha"] if ("alpha" in kwargs) else 0  # coefficient for adaptively learning beta, alpha=0 means using fixed beta
        self.vpt = kwargs["vpt"] if ("vpt" in kwargs) else 0
        self.atanh_power = kwargs["atanh_power"] if ("atanh_power" in kwargs) else 3

        self.use_prior_only = kwargs["use_prior_only"] if ("use_prior_only" in kwargs) else False
        # self.policy_use_prior = kwargs["policy_prior_only"] if ("policy_prior_only" in kwargs) else False

        self.tau = kwargs["tau"] if ("tau" in kwargs) else 500

        self.alg_config = kwargs["alg_config"] if ("alg_config" in kwargs) else {}
        self.lr = kwargs["lr"] if ("lr" in kwargs) else 3e-4

        self.gamma = kwargs["gamma"] if ("gamma" in kwargs) else 0.99
        self.value_distribution = kwargs["value_distribution"] if ("value_distribution" in kwargs) else "DiracDelta"

        self.batch_norm_tau = kwargs["batch_norm_tau"] if ("batch_norm_tau" in kwargs) else 0
        self.use_equal_net = kwargs["use_equal_net"] if ("use_equal_net" in kwargs) else True

        if "device" in kwargs:
            self.device = kwargs["device"]
        else:
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'

        self.verbose = kwargs["verbose"] if ("verbose" in kwargs) else 1

        self.recent_loss_set_size = kwargs["recent_loss_set_size"] if ("recent_loss_set_size" in kwargs) else 1000
        self.recent_loss_set = np.zeros(self.recent_loss_set_size)
        self.recent_loss_prior_set = np.zeros(self.recent_loss_set_size)

        self.update_times = 0

        self.z_size = self.z_stochastic_size + self.z_deterministic_size

        if self.beta:
            if self.alpha:
                log_beta = torch.tensor(np.log(self.beta).astype(np.float32)).to(device=self.device)
                self.log_beta = log_beta.clone().detach().requires_grad_(True)
            else:
                self.log_beta = torch.tensor(np.log(self.beta).astype(np.float32)).to(device=self.device)

        if self.act_fn == "relu":
            self.forward_act_fn = nn.ReLU
        elif self.act_fn == "tanh":
            self.forward_act_fn = nn.Tanh
        else:
            raise ValueError("activation function must be tanh or relu")

        # -------------------- Define Network Connections ------------------
        self.latent_module = nn.ModuleList()

        if self.use_equal_net:
            self.encoder = MahjongEqualNet(self.input_forward_size[0], batch_norm_tau=self.batch_norm_tau)
            self.encoder_oracle = MahjongEqualNet(self.input_oracle_size[0], batch_norm_tau=self.batch_norm_tau)
        else:
            self.encoder = MahjongNet(self.input_forward_size[0], batch_norm_tau=self.batch_norm_tau)
            self.encoder_oracle = MahjongNet(self.input_oracle_size[0], batch_norm_tau=self.batch_norm_tau)

        self.phi_size = self.encoder.phi_size
        self.phi_size_oracle = self.encoder_oracle.phi_size

        self.latent_module.append(self.encoder_oracle)
        self.latent_module.append(self.encoder)

        forward_fnns = nn.ModuleList()
        last_layer_size = self.phi_size
        for _ in range(self.half_hidden_layer_depth - 1):
            forward_fnns.append(nn.Linear(last_layer_size, self.hidden_layer_width))
            if self.batch_norm_tau:
                forward_fnns.append(BatchNorm1d(num_features=self.hidden_layer_width, momentum=1 / self.batch_norm_tau))
            forward_fnns.append(self.forward_act_fn())
            last_layer_size = self.hidden_layer_width
        self.forward_fnn = nn.Sequential(*forward_fnns)
        self.latent_module.append(self.forward_fnn)

        pre_zp_size = self.hidden_layer_width

        if not self.z_size == 0:
            if self.z_deterministic_size:
                if self.batch_norm_tau:
                    self.f_h2zp_det = nn.Sequential(nn.Linear(pre_zp_size, self.z_deterministic_size),
                                                    BatchNorm1d(num_features=self.z_deterministic_size,
                                                                   momentum=1 / self.batch_norm_tau),
                                                    )
                else:
                    self.f_h2zp_det = nn.Sequential(nn.Linear(pre_zp_size, self.z_deterministic_size)
                                                    )
                self.latent_module.append(self.f_h2zp_det)
            if self.z_stochastic_size:

                if self.batch_norm_tau:
                    self.f_h2muzp = nn.Sequential(nn.Linear(pre_zp_size, self.z_stochastic_size),
                                                  BatchNorm1d(num_features=self.z_stochastic_size,
                                                                 momentum=1 / self.batch_norm_tau)
                                                  )

                    self.f_h2logsigzp = nn.Sequential(nn.Linear(pre_zp_size, self.z_stochastic_size),
                                                      BatchNorm1d(num_features=self.z_stochastic_size,
                                                                     momentum=1 / self.batch_norm_tau),
                                                      MinusOneModule()
                                                      )
                else:
                    self.f_h2muzp = nn.Sequential(nn.Linear(pre_zp_size, self.z_stochastic_size)
                                                  )

                    self.f_h2logsigzp = nn.Sequential(nn.Linear(pre_zp_size, self.z_stochastic_size),
                                                      MinusOneModule()
                                                      )

                self.latent_module.append(self.f_h2logsigzp)
                self.latent_module.append(self.f_h2muzp)

        oracle_fnns = nn.ModuleList()
        last_layer_size = self.phi_size_oracle
        for _ in range(self.half_hidden_layer_depth - 1):
            oracle_fnns.append(nn.Linear(last_layer_size, self.hidden_layer_width))
            if self.batch_norm_tau:
                oracle_fnns.append(BatchNorm1d(num_features=self.hidden_layer_width, momentum=1 / self.batch_norm_tau))
            oracle_fnns.append(self.forward_act_fn())
            last_layer_size = self.hidden_layer_width

        self.oracle_fnn = nn.Sequential(*oracle_fnns)
        self.latent_module.append(self.oracle_fnn)

        if not self.z_size == 0:
            pre_zq_size = self.hidden_layer_width

            if self.z_deterministic_size:
                if self.batch_norm_tau:
                    self.f_hb2zq_det = nn.Sequential(nn.Linear(pre_zq_size, self.z_deterministic_size),
                                                     BatchNorm1d(num_features=self.z_deterministic_size, momentum=1 / self.batch_norm_tau)
                                                     )
                else:
                    self.f_hb2zq_det = nn.Sequential(nn.Linear(pre_zq_size, self.z_deterministic_size)
                                                     )
                self.latent_module.append(self.f_hb2zq_det)

            if self.z_stochastic_size:
                if self.batch_norm_tau:
                    self.f_hb2muzq = nn.Sequential(nn.Linear(pre_zq_size, self.z_stochastic_size),
                                                   BatchNorm1d(num_features=self.z_stochastic_size, momentum=1 / self.batch_norm_tau)
                                                   )

                    self.f_hb2logsigzq = nn.Sequential(nn.Linear(pre_zq_size, self.z_stochastic_size),
                                                       BatchNorm1d(num_features=self.z_stochastic_size, momentum=1 / self.batch_norm_tau),
                                                       MinusOneModule()
                                                       )
                else:
                    self.f_hb2muzq = nn.Sequential(nn.Linear(pre_zq_size, self.z_stochastic_size)
                                                   )

                    self.f_hb2logsigzq = nn.Sequential(nn.Linear(pre_zq_size, self.z_stochastic_size),
                                                       MinusOneModule()
                                                       )
                self.latent_module.append(self.f_hb2logsigzq)
                self.latent_module.append(self.f_hb2muzq)

        # ----------------------------- RL part -----------------------------

        pre_rl_size = self.z_size

        if self.algorithm == 'ddqn':

            self.epsilon = kwargs["epsilon"] if ("epsilon" in kwargs) else 0.05
            self.use_pql = self.alg_config["use_pql"] if ("use_pql" in self.alg_config) else False  # Peng's Q(lambda)
            self.cql_alpha = self.alg_config["cql_alpha"] if ("cql_alpha" in self.alg_config) else 0  # Conservative Q-learning (H)
            self.use_cql = True if self.cql_alpha else False
            # self.n_steps_return = self.alg_config["n_steps_return"] if ("n_steps_return" in self.alg_config) else False
            # self.lambd = self.alg_config["lambd"] if ("lambd" in self.alg_config) else 0.7  # Peng's Q(lambda)

            self.soft_update_target_network = self.alg_config["soft_update_target_network"] if (
                    "soft_update_target_network" in self.alg_config) else False
            self.dueling = self.alg_config["dueling"] if ("dueling" in self.alg_config) else True

            self.alg_type = 'value_based'

            # Q network 1
            self.f_s2q1 = DiscreteActionQNetwork(pre_rl_size, self.action_size, batch_norm_tau=self.batch_norm_tau,
                                                 hidden_layers=[self.hidden_layer_width] * self.half_hidden_layer_depth,
                                                 dueling=self.dueling, output_distribution=self.value_distribution,
                                                 act_fn=self.forward_act_fn)

            self.optimizer_q = torch.optim.Adam(self.parameters(), lr=self.lr)

        elif self.algorithm == 'bc':
            self.dropout = self.alg_config["dropout"] if ("dropout" in self.alg_config) else 0
            self.f_s2pi0 = DiscreteActionPolicyNetwork(pre_rl_size, self.action_size, batch_norm_tau=self.batch_norm_tau,
                                                       hidden_layers=[self.hidden_layer_width] * self.half_hidden_layer_depth,
                                                       act_fn=self.forward_act_fn, device=self.device)
            self.alg_type = 'supervised'
            self.ce_loss = nn.CrossEntropyLoss(reduction='none')
            self.optimizer_a = torch.optim.Adam(self.parameters(), lr=self.lr)

        else:
            raise NotImplementedError("algorithm can only be 'bc' or 'ddqn'")

        self.mse_loss = nn.MSELoss()

        if (not self.use_prior_only):
            self.optimizer_b = torch.optim.SGD([self.log_beta], lr=self.lr)

        self.to(device=self.device)

        self.h_t = None
        self.a_tm1 = None

        self.zp_tm1 = torch.zeros([1, self.z_deterministic_size + self.z_stochastic_size],
                                  dtype=torch.float32).to(device=self.device)

        # target network
        if is_main_network:

            target_net = DDQN(observation_space, full_observation_space, action_space, is_main_network=False, **kwargs)

            # synchronizing target network and main network
            state_dict_tar = target_net.state_dict()
            state_dict = self.state_dict()
            for key in list(target_net.state_dict().keys()):
                state_dict_tar[key] = state_dict[key]
            target_net.load_state_dict(state_dict_tar)

            self.target_net = target_net

        self.eval()

    def init_states(self):
        pass

    def select(self, x, action_mask=None, greedy=False, need_other_info=False):

        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x.astype(np.float32).reshape([1, *list(x.shape)])).to(device=self.device)
            x = self.encoder(x)

            if action_mask is not None:
                if isinstance(action_mask, np.ndarray):
                    action_mask = torch.from_numpy(
                        action_mask.astype(np.float32).reshape([1, self.action_size]))

            e = self.forward_fnn(x)
            if self.z_stochastic_size > 0:
                muz = self.f_h2muzp(e)
                logsigz = self.f_h2logsigzp(e)
                dist = dis.normal.Normal(muz, torch.exp(logsigz))
                z = dist.sample()

            elif self.z_deterministic_size > 0:
                z = self.f_h2zp_det(e)

            self.h_t = z

            if self.algorithm == 'bc':

                a = self.f_s2pi0.sample_action(self.h_t, action_mask=action_mask, greedy=greedy).item()

            elif self.algorithm == 'ddqn':
                q = self.f_s2q1(self.h_t).detach().cpu()
                if action_mask is not None:
                    q = q * action_mask - INFINITY * (1 - action_mask)

                if greedy:
                    a = torch.argmax(q, dim=-1)
                    if np.prod(a.shape) == 1:
                        a = a.item()  # discrete action
                else:
                    if np.random.rand() < self.epsilon:
                        if action_mask is None:
                            a = np.random.randint(self.action_size)
                        else:
                            valid_action_ind = np.nonzero(action_mask.cpu().numpy().reshape([-1]))
                            valid_actions = np.arange(self.action_size)[valid_action_ind]
                            a = valid_actions[np.random.randint(len(valid_actions))]
                    else:
                        a = torch.argmax(q, dim=-1)
                        if np.prod(a.shape) == 1:
                            a = a.item()  # discrete action

            self.a_tm1 = torch.from_numpy(np.array([a]).reshape([-1])).to(device=self.device)

        if not need_other_info:
            return a
        else:
            return a, self.h_t, self.zp_tm1

    def learn_bc(self, X, O, A, V, action_masks=None, full_obs=False, mahjong_augment=False):
        batch_size = A.shape[0]
        start_time = time.time()

        x_oracle = torch.from_numpy(np.concatenate([X, O], axis=-2).astype(np.float32)).to(device=self.device)

        if not full_obs:
            x = torch.from_numpy(X.astype(np.float32)).to(device=self.device)
        else:
            x = x_oracle

        if mahjong_augment:
            all_obs_concat = augment_mahjong_data(torch.cat([x, x_oracle], dim=-2))
            x = all_obs_concat[:, :x.shape[-2], :]
            x_oracle = all_obs_concat[:, -x_oracle.shape[-2]:, :]

        if action_masks is not None:
            m = torch.from_numpy(action_masks.astype(np.float32)).to(device=self.device)

        self.train()

        a = torch.from_numpy(A.astype(np.int)).to(device=self.device)
        v = torch.from_numpy(V.astype(np.float32)).to(device=self.device)

        phi = self.encoder(x)
        phi_oracle = self.encoder_oracle(x_oracle)

        if not self.use_prior_only:

            if self.z_stochastic_size > 0 and self.z_deterministic_size == 0:

                e = self.forward_fnn(phi)
                e_oracle = self.oracle_fnn(phi_oracle)

                muzp_tensor = self.f_h2muzp(e)
                logsigzp_tensor = self.f_h2logsigzp(e)

                muzq_tensor = self.f_hb2muzq(e_oracle)
                logsigzq_tensor = self.f_hb2logsigzq(e_oracle)

                dist_q = dis.normal.Normal(muzq_tensor, torch.exp(logsigzq_tensor))
                h_tensor = dist_q.rsample()
                zq_tensor = h_tensor

                dist_p = dis.normal.Normal(muzp_tensor, torch.exp(logsigzp_tensor))
                hp_tensor = dist_p.rsample()
                zp_tensor = hp_tensor

            elif self.z_deterministic_size > 0 and self.z_stochastic_size == 0:
                self.train()

                zp_tensor = self.f_h2zp_det(self.forward_fnn(phi))
                zq_tensor = self.f_hb2zq_det(self.oracle_fnn(phi_oracle))

                h_tensor = zq_tensor
                hp_tensor = zp_tensor

            else:
                raise NotImplementedError
        else:
            # use_prior_only (baseline)
            if self.z_stochastic_size > 0 and self.z_deterministic_size == 0:
                e_t = self.forward_fnn(phi)
                muzp_tensor = self.f_h2muzp(e_t)
                logsigzp_tensor = self.f_h2logsigzp(self.forward_fnn(e_t))
                dist_p = dis.normal.Normal(muzp_tensor, torch.exp(logsigzp_tensor))
                zp_tensor = dist_p.rsample()
                h_tensor = zp_tensor

            elif self.z_deterministic_size > 0 and self.z_stochastic_size == 0:
                zp_tensor = self.f_h2zp_det(self.forward_fnn(phi))
                h_tensor = zp_tensor
            else:
                raise NotImplementedError

        # ------------------ loss z -------------------
        if not self.use_prior_only:
            if self.z_stochastic_size == 0 and self.z_deterministic_size > 0:

                loss_z = 1. / self.action_size * torch.mean(
                    torch.sum(0.5 * (zq_tensor - zp_tensor).pow(2), dim=-1) * v)  # deterministic z only

                if self.verbose and np.random.rand() < 0.005 * self.verbose:
                    tmp = np.array2string(zp_tensor[0, :6].detach().cpu().numpy(), precision=4, separator=',  ')
                    print("zp = " + tmp)
                    tmp = np.array2string(zq_tensor[0, :6].detach().cpu().numpy(), precision=4, separator=',  ')
                    print("zq = " + tmp)

            elif self.z_deterministic_size == 0 and self.z_stochastic_size > 0:

                if self.verbose and np.random.rand() < 0.005 * self.verbose:
                    tmp = np.array2string(muzp_tensor[0, :6].detach().cpu().numpy(), precision=4, separator=',  ')
                    print("muzp = " + tmp)
                    tmp = np.array2string(muzq_tensor[0, :6].detach().cpu().numpy(), precision=4, separator=',  ')
                    print("muzq = " + tmp)

                    tmp = np.array2string(logsigzp_tensor[0, :6].detach().cpu().exp().numpy(),
                                          precision=4, separator=', ')
                    print("sigzp = " + tmp)
                    tmp = np.array2string(logsigzq_tensor[0, :6].detach().cpu().exp().numpy(),
                                          precision=4, separator=', ')
                    print("sigzq = " + tmp)

                loss_z = torch.mean(
                    torch.sum(logsigzp_tensor - logsigzq_tensor + ((muzp_tensor - muzq_tensor).pow(2) + torch.exp(
                        logsigzq_tensor * 2)) / (2.0 * torch.exp(logsigzp_tensor * 2)) - 0.5, dim=-1) * v)

                loss_z = 1. / self.action_size * loss_z
            else:
                raise NotImplementedError
        else:
            loss_z = torch.tensor(0)

        if (not self.use_prior_only) and (self.kld_target >= 0) and torch.sum(v).item() > 0:
            kld = (loss_z * self.action_size).data

        a_predict = self.f_s2pi0(h_tensor)
        loss_a = torch.mean(self.ce_loss(a_predict, a) * v)
        loss_critic = torch.tensor(0)

        self.optimizer_a.zero_grad()
        if not self.use_prior_only:
            (torch.exp(self.log_beta.detach()) * loss_z + loss_a).backward()
        else:
            loss_a.backward()
        self.optimizer_a.step()

        if (not self.use_prior_only) and (self.kld_target >= 0) and torch.sum(v).item() > 0:
            loss_beta = - torch.mean(self.log_beta * self.alpha * (
                    torch.log10(torch.clamp(kld, 1e-9, np.inf)) - np.log10(self.kld_target)).detach())

            if self.verbose and np.random.rand() < 0.005 * self.verbose:
                print("log beta = {}, kld = {}, kld_target= {}".format(
                    self.log_beta.item(), kld.detach().item(), self.kld_target))

            self.optimizer_b.zero_grad()
            loss_beta.backward()
            self.optimizer_b.step()

        self.eval()

        if self.update_times < 10:
            print("training time:", time.time() - start_time)

        self.update_times += 1

        return loss_z.cpu().item(), loss_critic.cpu().item(), loss_a.cpu().item()

    def learn(self, X, XP, O, OP, A, R, D, V, action_masks=None, action_masks_tp1=None, full_obs=False, mahjong_augment=False):

        batch_size = A.shape[0]
        start_time = time.time()

        x_oracle = torch.from_numpy(np.concatenate([X, O], axis=-2).astype(np.float32)).to(device=self.device)
        xp_oracle = torch.from_numpy(np.concatenate([XP, OP], axis=-2).astype(np.float32)).to(device=self.device)

        if not full_obs:
            x = torch.from_numpy(X.astype(np.float32)).to(device=self.device)
            xp = torch.from_numpy(XP.astype(np.float32)).to(device=self.device)
        else:
            x = x_oracle
            xp = xp_oracle

        if mahjong_augment:
            all_obs_concat = augment_mahjong_data(torch.cat([x, xp, x_oracle, xp_oracle], dim=-2))

            x = all_obs_concat[:, :x.shape[-2], :]
            xp = all_obs_concat[:, x.shape[-2]: int(2 * x.shape[-2]), :]
            x_oracle = all_obs_concat[:, int(2 * x.shape[-2]): int(2 * x.shape[-2] + x_oracle.shape[-2]), :]
            xp_oracle = all_obs_concat[:, int(2 * x.shape[-2] + x_oracle.shape[-2]):, :]

        a = torch.from_numpy(A.astype(np.int)).to(device=self.device)
        r = torch.from_numpy(R.astype(np.float32)).to(device=self.device)
        d = torch.from_numpy(D.astype(np.float32)).to(device=self.device)
        v = torch.from_numpy(V.astype(np.float32)).to(device=self.device)

        if action_masks is not None:
            m = torch.from_numpy(action_masks.astype(np.float32)).to(device=self.device)
            mp = torch.from_numpy(action_masks_tp1.astype(np.float32)).to(device=self.device)

        # print(x.shape)

        self.train()
        self.target_net.eval()

        phi = self.encoder(x)
        phi_oracle = self.encoder_oracle(x_oracle)

        phi_tp1 = self.encoder(xp)
        phi_oracle_tp1 = self.encoder_oracle(xp_oracle)

        # phi_tar = self.target_net.encoder(x)
        # phi_oracle_tar = self.target_net.encoder(x_oracle)

        phi_tp1_tar = self.target_net.encoder(xp)
        phi_oracle_tp1_tar = self.target_net.encoder_oracle(xp_oracle)

        if not self.use_prior_only:

            if self.z_stochastic_size > 0 and self.z_deterministic_size == 0:

                self.train()

                e = self.forward_fnn(phi)
                e_oracle = self.oracle_fnn(phi_oracle)

                muzp_tensor = self.f_h2muzp(e)
                logsigzp_tensor = self.f_h2logsigzp(e)

                muzq_tensor = self.f_hb2muzq(e_oracle)
                logsigzq_tensor = self.f_hb2logsigzq(e_oracle)

                dist_q = dis.normal.Normal(muzq_tensor, torch.exp(logsigzq_tensor))
                h_tensor = dist_q.rsample()
                zq_tensor = h_tensor

                dist_p = dis.normal.Normal(muzp_tensor, torch.exp(logsigzp_tensor))
                hp_tensor = dist_p.rsample()
                zp_tensor = hp_tensor

                with torch.no_grad():
                    self.eval()
                    e_oracle_tp1 = self.oracle_fnn(phi_oracle_tp1)

                    muzq_tensor_tp1 = self.f_hb2muzq(e_oracle_tp1)
                    logsigzq_tensor_tp1 = self.f_hb2logsigzq(e_oracle_tp1)

                    dist = dis.normal.Normal(muzq_tensor_tp1, torch.exp(logsigzq_tensor_tp1))
                    h_tp1_tensor = dist.sample().detach()

                    e_oracle_tp1_tar = self.target_net.oracle_fnn(phi_oracle_tp1_tar)

                    muzq_tensor_tp1_tar = self.target_net.f_hb2muzq(e_oracle_tp1_tar)
                    logsigzq_tensor_tp1_tar = self.target_net.f_hb2logsigzq(e_oracle_tp1_tar)

                    dist = dis.normal.Normal(muzq_tensor_tp1_tar, torch.exp(logsigzq_tensor_tp1_tar))
                    h_tp1_tar_tensor = dist.sample().detach()

            elif self.z_deterministic_size > 0 and self.z_stochastic_size == 0:
                self.train()

                zp_tensor = self.f_h2zp_det(self.forward_fnn(phi))
                zq_tensor = self.f_hb2zq_det(self.oracle_fnn(phi_oracle))

                h_tensor = zq_tensor
                hp_tensor = zp_tensor

                with torch.no_grad():
                    self.eval()
                    e_oracle_tp1 = self.oracle_fnn(phi_oracle_tp1).detach_()
                    h_tp1_tensor = self.f_hb2zq_det(e_oracle_tp1).detach_()

                    e_oracle_tp1_tar = self.target_net.oracle_fnn(phi_oracle_tp1_tar).detach_()
                    h_tp1_tar_tensor = self.target_net.f_hb2zq_det(e_oracle_tp1_tar).detach_()
            else:
                raise NotImplementedError
        else:
            # use_prior_only (baseline)
            if self.z_stochastic_size > 0 and self.z_deterministic_size == 0:
                e_t = self.forward_fnn(phi)
                muzp_tensor = self.f_h2muzp(e_t)
                logsigzp_tensor = self.f_h2logsigzp(self.forward_fnn(e_t))
                dist_p = dis.normal.Normal(muzp_tensor, torch.exp(logsigzp_tensor))
                zp_tensor = dist_p.rsample()

                h_tensor = zp_tensor

                with torch.no_grad():
                    self.eval()
                    e_tp1 = self.forward_fnn(phi_tp1).detach()

                    muzp_tensor_tp1 = self.f_h2muzp(e_tp1)
                    logsigzp_tensor_tp1 = self.f_h2logsigzp(self.forward_fnn(e_tp1))
                    dist_p = dis.normal.Normal(muzp_tensor_tp1, torch.exp(logsigzp_tensor_tp1))
                    h_tp1_tensor = dist_p.sample().detach()

                    e_tp1_tar = self.target_net.forward_fnn(phi_tp1_tar).detach()
                    muzp_tensor_tp1_tar = self.f_h2muzp(e_tp1_tar)
                    logsigzp_tensor_tp1_tar = self.f_h2logsigzp(self.forward_fnn(e_tp1_tar))
                    dist_p = dis.normal.Normal(muzp_tensor_tp1_tar, torch.exp(logsigzp_tensor_tp1_tar))
                    h_tp1_tar_tensor = dist_p.sample().detach()

            elif self.z_deterministic_size > 0 and self.z_stochastic_size == 0:
                zp_tensor = self.f_h2zp_det(self.forward_fnn(phi))
                h_tensor = zp_tensor

                with torch.no_grad():
                    self.eval()
                    e_tp1 = self.forward_fnn(phi_tp1).detach()
                    h_tp1_tensor = self.f_h2zp_det(e_tp1).detach()

                    e_tp1_tar = self.target_net.forward_fnn(phi_tp1_tar).detach()
                    h_tp1_tar_tensor = self.target_net.f_h2zp_det(e_tp1_tar).detach()
            else:
                raise NotImplementedError

        # if self.alg_type is 'actor_critic':
        #     phi_critic = self.encoder_critic(x_oracle)
        #     phi_critic_tar = self.target_net.encoder_critic(x_oracle).detach()
        #     phi_tp1_critic = self.encoder_critic(xp_oracle).detach()
        #     phi_tp1_critic_tar = self.target_net.encoder_critic(xp_oracle).detach()

        # ------------ compute divergence between z^p and z^q ------------
        self.train()

        if not self.use_prior_only:
            if self.z_stochastic_size == 0 and self.z_deterministic_size > 0:

                loss_z = 1. / self.action_size * torch.mean(
                    torch.sum(0.5 * (zq_tensor - zp_tensor).pow(2), dim=-1) * v)  # deterministic z only

                if self.verbose and np.random.rand() < 0.005 * self.verbose:
                    tmp = np.array2string(zp_tensor[0, :6].detach().cpu().numpy(), precision=4, separator=',  ')
                    print("zp = " + tmp)
                    tmp = np.array2string(zq_tensor[0, :6].detach().cpu().numpy(), precision=4, separator=',  ')
                    print("zq = " + tmp)

            elif self.z_deterministic_size == 0 and self.z_stochastic_size > 0:

                if self.verbose and np.random.rand() < 0.005 * self.verbose:
                    tmp = np.array2string(muzp_tensor[0, :6].detach().cpu().numpy(), precision=4, separator=',  ')
                    print("muzp = " + tmp)
                    tmp = np.array2string(muzq_tensor[0, :6].detach().cpu().numpy(), precision=4, separator=',  ')
                    print("muzq = " + tmp)

                    tmp = np.array2string(logsigzp_tensor[0, :6].detach().cpu().exp().numpy(),
                                          precision=4, separator=', ')
                    print("sigzp = " + tmp)
                    tmp = np.array2string(logsigzq_tensor[0, :6].detach().cpu().exp().numpy(),
                                          precision=4, separator=', ')
                    print("sigzq = " + tmp)

                loss_z = torch.mean(
                    torch.sum(logsigzp_tensor - logsigzq_tensor + ((muzp_tensor - muzq_tensor).pow(2) + torch.exp(
                        logsigzq_tensor * 2)) / (2.0 * torch.exp(logsigzp_tensor * 2)) - 0.5, dim=-1) * v)

                loss_z = 1. / self.action_size * loss_z
            else:
                raise NotImplementedError
        else:
            loss_z = torch.tensor(0)

        self.eval()
        # ------------ compute (posterior) value prediction loss  -------------
        if self.algorithm == 'ddqn':

            gamma = self.gamma

            # ---------- Compute Q Target ---------------
            self.eval()

            # q_tensor = self.f_s2q1(h_tensor)
            qp_tensor = self.f_s2q1(h_tp1_tensor).detach()
            qp_tensor_tar = self.target_net.f_s2q1(h_tp1_tar_tensor).detach()

            if action_masks is not None:
                qp_tensor = qp_tensor * mp - (1 - mp) * INFINITY
                qp_tensor_tar = qp_tensor_tar * mp - (1 - mp) * INFINITY

            a_one_hot = F.one_hot(a.to(torch.int64), num_classes=self.action_size).to(torch.float32)

            # double Q-learning
            a_greedy = torch.argmax(qp_tensor, dim=-1, keepdim=False)
            # greedy action is selected using the main network
            a_greedy_one_hot = F.one_hot(a_greedy.to(torch.int64), num_classes=self.action_size).to(torch.float32)
            q_target = (r + (1 - d) * gamma * torch.sum(qp_tensor_tar.detach() * a_greedy_one_hot, dim=-1))

            q_target_expand = q_target.view([*q_target.size(), 1]).repeat_interleave(self.action_size, dim=-1)

            # ---------- Train ----------------
            self.train()
            self.target_net.eval()

            loss_critic = torch.mean(
                torch.sum(- self.f_s2q1.get_log_prob(h_tensor, q_target_expand) * a_one_hot, dim=-1) * v)

            self.eval()

            if self.use_cql:
                loss_cql = self.cql_alpha * torch.mean(torch.logsumexp(self.f_s2q1(h_tensor), dim=-1)
                                                       - torch.sum(a_one_hot * self.f_s2q1(h_tensor), dim=-1))
            else:
                loss_cql = 0

            # loss_critic = 0.5 * self.mse_loss(torch.sum(q_tensor * a_one_hot, dim=-1) * v, q_target * v)

            if (not self.use_prior_only) and (self.kld_target >= 0) and torch.sum(v).item() > 0:
                kld = (loss_z * self.action_size).data

            elif (not self.use_prior_only) and self.alpha and torch.sum(v).item() > 0:
                q_prior_expanded = self.f_s2q1(hp_tensor).detach()
                q_posterior_expanded = self.f_s2q1(h_tensor).detach()

                diff_vpq = torch.sqrt(torch.mean(
                    (torch.square(q_posterior_expanded - q_prior_expanded) * a_one_hot).sum(dim=-1) * v).detach())
                diff_vqt = torch.sqrt(torch.mean(
                    (torch.square(q_posterior_expanded - q_target_expand) * a_one_hot).sum(dim=-1) * v).detach())
                diff_vpt = torch.sqrt(torch.mean(
                    (torch.square(q_target_expand - q_prior_expanded) * a_one_hot).sum(dim=-1) * v).detach())

                if self.verbose and np.random.rand() < 0.005 * self.verbose:
                    print("log beta = {}, diff_qpq = {}, diff_qtar= {}".format(
                        self.log_beta.item(), diff_vpq.detach().item(), diff_vqt.detach().item()))

            self.optimizer_q.zero_grad()

            if not self.use_prior_only:
                (torch.exp(self.log_beta.detach()) * loss_z + loss_critic + loss_cql).backward()  # 2 * loss_z because value and policy both have KLD loss
            else:
                (loss_critic + loss_cql).backward()

            self.optimizer_q.step()

            loss_a = torch.tensor(0)

            if (not self.use_prior_only) and (self.kld_target >= 0) and torch.sum(v).item() > 0:
                loss_beta = - torch.mean(self.log_beta * self.alpha * (
                        torch.log10(torch.clamp(kld, 1e-9, np.inf)) - np.log10(self.kld_target)).detach())

                if self.verbose and np.random.rand() < 0.005 * self.verbose:
                    print("log beta = {}, kld = {}, kld_target= {}".format(
                        self.log_beta.item(), kld.detach().item(), self.kld_target))

                self.optimizer_b.zero_grad()
                loss_beta.backward()
                self.optimizer_b.step()

            elif (not self.use_prior_only) and self.alpha and torch.sum(v).item() > 0:

                if not self.vpt:
                    loss_diff = (self.alpha * diff_vpq.detach() - diff_vqt.detach()) / \
                                (self.alpha * diff_vpq.detach() + diff_vqt.detach())

                else:
                    loss_diff = (self.alpha * diff_vpq.detach() - diff_vpt.detach()) / \
                                (self.alpha * diff_vpq.detach() + diff_vpt.detach())

                loss_beta = - torch.mean(self.log_beta * torch.sign(loss_diff) * (torch.abs(torch.atanh(loss_diff)) ** self.atanh_power))

                self.optimizer_b.zero_grad()
                loss_beta.backward()
                self.optimizer_b.step()

            # update target Q network
            if self.soft_update_target_network:
                state_dict_tar = self.target_net.state_dict()
                state_dict = self.state_dict()
                for key in list(self.target_net.state_dict().keys()):
                    state_dict_tar[key] = (1 - 1 / self.tau) * state_dict_tar[key] + 1 / self.tau * state_dict[key]
                    # state_dict_tar[key] = 0 * state_dict_tar[key] + 1 * state_dict[key]
                self.target_net.load_state_dict(state_dict_tar)

            else:
                if self.update_times % int(self.tau) == 0:
                    state_dict_tar = self.target_net.state_dict()
                    state_dict = self.state_dict()
                    for key in list(self.target_net.state_dict().keys()):
                        state_dict_tar[key] = 0 * state_dict_tar[key] + 1 * state_dict[key]
                    self.target_net.load_state_dict(state_dict_tar)

        # --------------- end ddqn -----------------


        if self.update_times < 10:
            print("training time:", time.time() - start_time)

        self.update_times += 1

        return loss_z.cpu().item(), loss_critic.cpu().item(), loss_a.cpu().item()
