import warnings

import gym
import torch
import torch.nn as nn
import numpy as np
import time
from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributions as dis
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from gym.spaces import Box, Discrete

from base_modules import *

torch.set_default_dtype(torch.float32)

INFINITY = 1e10


def init_weights_kaiming(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


def init_weights_xavier(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)


def init_weights_uniform(m):
    if type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, a=-0.15, b=0.15)


def init_bias_uniform(m):
    if type(m) == nn.Linear:
        torch.nn.init.uniform_(m.bias, a=-3, b=3)


def softplus_exp(x):
    y = torch.log(1 + torch.exp(x))
    return y


def compute_kernel(x, y, v):
    if len(x.shape) == 3:
        v = v.view([x.shape[0] * x.shape[1]])
        x = x.view([-1, x.shape[-1]])
        y = y.view([-1, y.shape[-1]])
    x_size = x.shape[0]
    y_size = y.shape[0]
    dim = x.shape[1]

    tiled_x = x.view(x_size, 1, dim).repeat(1, y_size, 1)
    tiled_y = y.view(1, y_size, dim).repeat(x_size, 1, 1)
    tiled_v = v.view([1, y_size]).repeat(x_size, 1)

    return torch.exp(-torch.mean((tiled_x - tiled_y) ** 2, dim=2) / dim * 1.0) * tiled_v


def compute_mmd(mux, muy, v):
    x_kernel = compute_kernel(mux, mux, v)
    y_kernel = compute_kernel(muy, muy, v)
    xy_kernel = compute_kernel(mux, muy, v)
    return torch.mean(x_kernel + y_kernel - 2 * xy_kernel)


# def compute_kernel(mux, logsigx, muy, logsigy, v):
#     assert mux.shape == logsigx.shape == muy.shape == logsigy.shape
#
#     if len(mux.shape) == 3:
#         v = v.view([mux.shape[0] * mux.shape[1]])
#         mux = mux.view([-1, mux.shape[-1]])
#         muy = muy.view([-1, muy.shape[-1]])
#         logsigx = logsigx.view([-1, logsigx.shape[-1]])
#         logsigy = logsigy.view([-1, logsigy.shape[-1]])
#
#     x_size = mux.shape[0]
#     y_size = muy.shape[0]
#     dim = mux.shape[-1]
#
#     tiled_mux = mux.view([x_size, 1, dim]).repeat(1, y_size, 1)
#     tiled_muy = muy.view([1, y_size, dim]).repeat(x_size, 1, 1)
#     tiled_logsigx = logsigx.view([x_size, 1, dim]).repeat(1, y_size, 1)
#     tiled_logsigy = logsigy.view([1, y_size, dim]).repeat(x_size, 1, 1)
#     tiled_v = v.view([1, y_size]).repeat(x_size, 1)
#
#     kld = torch.sum(tiled_logsigx - tiled_logsigy + ((tiled_mux - tiled_muy).pow(2) + torch.exp(tiled_logsigy * 2)) / (
#             2.0 * torch.exp(tiled_logsigx * 2)) - 0.5, dim=-1, keepdim=False) * tiled_v
#
#     return kld


# def compute_mmd(mux, logsigx, muy, logsigy, v):
#     x_kernel = compute_kernel(mux, logsigx, mux, logsigx, v)
#     y_kernel = compute_kernel(muy, logsigy, muy, logsigy, v)
#     xy_kernel = compute_kernel(mux, logsigx, muy, logsigy, v)
#     return torch.mean(x_kernel + y_kernel - 2 * xy_kernel)


class FakeGRUCell(nn.Module):
    def __init__(self, rnn_input_size, hidden_size):

        # Used like GRUCell, but no recurrent connection

        super(FakeGRUCell, self).__init__()

        self.mlp = nn.Sequential(nn.Linear(rnn_input_size, hidden_size),
                                 nn.Tanh(),
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.Tanh()
                                 )

    def forward(self, rnn_input, hidden):

        return self.mlp(rnn_input)


class VLOG(nn.Module):

    def __init__(self, observation_space, full_observation_space, action_space, is_main_network=True, **kwargs):

        super(VLOG, self).__init__()

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

        self.h_forward_size = kwargs["h_forward_size"] if ("h_forward_size" in kwargs) else 256
        self.h_oracle_size = kwargs["h_oracle_size"] if ("h_oracle_size" in kwargs) else 256
        self.hidden_layer_width = kwargs["hidden_layer_width"] if ("hidden_layer_width" in kwargs) else 256
        self.half_hidden_layer_depth = kwargs["half_hidden_layer_depth"] if ("half_hidden_layer_depth" in kwargs) else 2  #only for FNN case
        self.act_fn = kwargs["act_fn"] if ("act_fn" in kwargs) else 'relu'

        self.mmd = kwargs["mmd"] if ("mmd" in kwargs) else 0  # whether to use Maximum Mean Discrepancy

        self.forward_use_rnn = kwargs["forward_use_rnn"] if ("forward_use_rnn" in kwargs) else True
        self.oracle_use_rnn = kwargs["oracle_use_rnn"] if ("oracle_use_rnn" in kwargs) else True

        self.z_stochastic_size = kwargs["z_stochastic_size"] if ("z_stochastic_size" in kwargs) else 32  # not used currently
        self.z_deterministic_size = kwargs["z_deterministic_size"] if ("z_deterministic_size" in kwargs) else 0

        self.beta = kwargs["beta"] if ("beta" in kwargs) else 1.0

        self.kld_target = kwargs["kld_target"] if ("kld_target" in kwargs) else -1
        self.pq_div_threshold = kwargs["pq_div_threshold"] if ("pq_div_threshold" in kwargs) else 0
        self.recent_loss_set_size = kwargs["recent_loss_set_size"] if ("recent_loss_set_size" in kwargs) else 1000
        self.recent_loss_set = np.zeros(self.recent_loss_set_size)
        self.recent_loss_prior_set = np.zeros(self.recent_loss_set_size)

        self.alpha = kwargs["alpha"] if ("alpha" in kwargs) else 0  # coefficient for adaptively learning beta, alpha=0 means using fixed beta
        self.vpt = kwargs["vpt"] if ("vpt" in kwargs) else 0
        self.atanh_power = kwargs["atanh_power"] if ("atanh_power" in kwargs) else 3

        self.action_to_zp = kwargs["action_to_zp"] if ("action_to_zp" in kwargs) else False
        self.action_to_zq = kwargs["action_to_zq"] if ("action_to_zq" in kwargs) else self.action_to_zp

        self.use_prior_only = kwargs["use_prior_only"] if ("use_prior_only" in kwargs) else False
        # self.policy_use_prior = kwargs["policy_prior_only"] if ("policy_prior_only" in kwargs) else False

        self.rnn_state_detach = kwargs["rnn_state_detach"] if ("rnn_state_detach" in kwargs) else False

        self.target_net_include_rnn = kwargs["target_net_include_rnn"] if ("target_net_include_rnn" in kwargs) else True
        self.tau = kwargs["tau"] if ("tau" in kwargs) else 500

        self.alg_config = kwargs["alg_config"] if ("alg_config" in kwargs) else {}
        self.lr = kwargs["lr"] if ("lr" in kwargs) else 3e-4

        self.gamma = kwargs["gamma"] if ("gamma" in kwargs) else 0.99
        self.value_distribution = kwargs["value_distribution"] if ("value_distribution" in kwargs) else "DiracDelta"
        self.device = kwargs["device"] if ("device" in kwargs) else 'cpu'
        self.verbose = kwargs["verbose"] if ("verbose" in kwargs) else 1

        self.update_times = 0

        self.weight_init_method = kwargs["weight_init_method"] if ("weight_init_method" in kwargs) else None

        self.z_size = self.z_stochastic_size + self.z_deterministic_size

        if self.beta:
            if self.alpha or self.pq_div_threshold:
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
        # forward RNN
        self.rnn_module = nn.ModuleList()

        if len(self.input_forward_size) in [2, 3]:
            # TODO: when oracle observation has different shape
            self.image_input = True

            if len(self.input_forward_size) == 2:
                resolution = "{}".format(self.input_forward_size[1])
            else:
                resolution = "{}x{}".format(self.input_forward_size[1], self.input_forward_size[2])

            self.encoder, phi_size = make_cnn(resolution, self.input_forward_size[0])

            self.encoder_oracle, phi_size_oracle = make_cnn(resolution, self.input_oracle_size[0])

            self.encoder_critic, phi_size_oracle = make_cnn(resolution, self.input_oracle_size[0])
            # only used for actor critic

            self.phi_size = int(phi_size)
            self.phi_size_oracle = int(phi_size_oracle)

            self.rnn_module.append(self.encoder)
            self.rnn_module.append(self.encoder_oracle)

        elif len(self.input_forward_size) == 1:
            self.image_input = False
            self.encoder = nn.Sequential(nn.Linear(np.sum(self.input_forward_size), self.hidden_layer_width),
                                         self.forward_act_fn()
                                         )
            self.encoder_oracle = nn.Sequential(nn.Linear(np.sum(self.input_oracle_size), self.hidden_layer_width),
                                                self.forward_act_fn()
                                                )
            self.encoder_critic = nn.Sequential(nn.Linear(np.sum(self.input_oracle_size), self.hidden_layer_width),
                                                self.forward_act_fn()
                                                )  # only used for actor critic

            self.rnn_module.append(self.encoder)
            self.rnn_module.append(self.encoder_oracle)
            self.phi_size = self.hidden_layer_width
            self.phi_size_oracle = self.hidden_layer_width
        else:
            raise NotImplementedError("Observation space must be either 2-D, 3-D (image-like), or 1-D (vector)!")

        if self.forward_use_rnn:

            if self.forward_use_rnn == -1:
                self.rnn_forward = nn.GRUCell(self.phi_size + self.z_deterministic_size + self.z_stochastic_size,
                                              self.h_forward_size)
            else:
                self.rnn_forward = FakeGRUCell(self.phi_size + self.z_deterministic_size + self.z_stochastic_size,
                                               self.h_forward_size)
            warnings.warn("Using FakeGRUCell !!!!!!!")
            self.rnn_module.append(self.rnn_forward)
        else:
            forward_fnns = nn.ModuleList()
            last_layer_size = self.phi_size
            for _ in range(self.half_hidden_layer_depth - 1):
                forward_fnns.append(nn.Linear(last_layer_size, self.h_forward_size))
                forward_fnns.append(self.forward_act_fn())
                last_layer_size = self.h_forward_size
            self.forward_fnn = nn.Sequential(*forward_fnns)

            # self.forward_fnn = nn.Sequential(nn.Linear(self.phi_size, self.h_forward_size),
            #                                  self.forward_act_fn(),
            #                                  )
            # self.forward_fnn = nn.Identity()
            self.rnn_module.append(self.forward_fnn)

        pre_zp_size = self.h_forward_size + self.action_size if (self.action_to_zp and self.forward_use_rnn) \
            else self.h_forward_size

        if not self.z_size == 0:
            if self.z_deterministic_size:
                self.f_h2zp_det = nn.Sequential(self.forward_act_fn(),
                                                nn.Linear(pre_zp_size, self.z_deterministic_size)
                                                )
                self.rnn_module.append(self.f_h2zp_det)
            if self.z_stochastic_size:
                self.f_h2muzp = nn.Sequential(self.forward_act_fn(),
                                              nn.Linear(pre_zp_size, self.z_stochastic_size)
                                              )
                self.rnn_module.append(self.f_h2muzp)

                self.f_h2logsigzp = nn.Sequential(self.forward_act_fn(),
                                                  nn.Linear(pre_zp_size, self.z_stochastic_size),
                                                  MinusOneModule()
                                                  )
                self.rnn_module.append(self.f_h2logsigzp)

        # oracle RNN (backward)
        if self.oracle_use_rnn:
            self.rnn_oracle = nn.GRU(self.phi_size_oracle, self.h_oracle_size, batch_first=True)
            self.rnn_module.append(self.rnn_oracle)
        else:
            oracle_fnns = nn.ModuleList()
            last_layer_size = self.phi_size_oracle
            for _ in range(self.half_hidden_layer_depth - 1):
                oracle_fnns.append(nn.Linear(last_layer_size, self.h_oracle_size))
                oracle_fnns.append(self.forward_act_fn())
                last_layer_size = self.h_oracle_size
            self.oracle_fnn = nn.Sequential(*oracle_fnns)

            # self.oracle_fnn = nn.Sequential(nn.Linear(self.phi_size_oracle, self.h_oracle_size),
            #                                 self.forward_act_fn(),
            #                                 )
            self.rnn_module.append(self.oracle_fnn)

        if not self.z_size == 0:
            if self.forward_use_rnn:
                pre_zq_size = self.h_oracle_size + self.h_forward_size
                if self.action_to_zq:
                    pre_zq_size += self.action_size
            else:
                pre_zq_size = self.h_oracle_size

            if self.z_deterministic_size:
                self.f_hb2zq_det = nn.Sequential(self.forward_act_fn(),
                                                 nn.Linear(pre_zq_size, self.z_deterministic_size)
                                                 )
                self.rnn_module.append(self.f_hb2zq_det)

            if self.z_stochastic_size:
                self.f_hb2muzq = nn.Sequential(self.forward_act_fn(),
                                               nn.Linear(pre_zq_size, self.z_stochastic_size)
                                               )
                self.rnn_module.append(self.f_hb2muzq)

                self.f_hb2logsigzq = nn.Sequential(self.forward_act_fn(),
                                                   nn.Linear(pre_zq_size, self.z_stochastic_size),
                                                   MinusOneModule()
                                                   )
                self.rnn_module.append(self.f_hb2logsigzq)

        # ----------------------------- RL part -----------------------------

        if self.forward_use_rnn:
            pre_rl_size = self.h_forward_size
        elif self.z_size > 0:
            pre_rl_size = self.z_size
        else:
            raise ValueError("Dimension of latent variable z cannot be 0 when using feedforward network")

        if self.algorithm == 'sac':

            self.target_entropy = np.float32(- self.action_size)
            self.alg_type = 'actor_critic'

            self.beta_h = kwargs["beta_h"] if ("beta_h" in kwargs) else 'auto_1.0'
            self.n_steps_return = self.alg_config["n_steps_return"] if ("n_steps_return" in self.alg_config) else 0

            if isinstance(self.beta_h, str) and self.beta_h.startswith('auto'):
                # Default initial value of beta_h when learned
                init_value = 1.0
                if '_' in self.beta_h:
                    init_value = float(self.beta_h.split('_')[1])
                    assert init_value > 0., "The initial value of beta_h must be greater than 0"
                self.log_beta_h = torch.tensor(np.log(init_value).astype(np.float32), requires_grad=True)
            else:
                # Force conversion to float
                # this will throw an error if a malformed string (different from 'auto')
                # is passed
                self.beta_h = float(self.beta_h)

            if isinstance(self.beta_h, str):
                self.optimizer_e = torch.optim.Adam([self.log_beta_h], lr=self.lr)  # optimizer for beta_h

            # policy network
            self.f_s2pi0 = ContinuousActionPolicyNetwork(pre_rl_size, self.action_size, act_fn=self.forward_act_fn,
                                                         hidden_layers=[self.hidden_layer_width] * self.half_hidden_layer_depth)

            # V network
            self.f_s2v = ContinuousActionVNetwork(self.phi_size_oracle, act_fn=self.forward_act_fn,
                                                  hidden_layers=[self.hidden_layer_width] * self.half_hidden_layer_depth)

            # Q network 1
            self.f_s2q1 = ContinuousActionQNetwork(self.phi_size_oracle, self.action_size, act_fn=self.forward_act_fn,
                                                   hidden_layers=[self.hidden_layer_width] * self.half_hidden_layer_depth)

            # Q network 2
            self.f_s2q2 = ContinuousActionQNetwork(self.phi_size_oracle, self.action_size, act_fn=self.forward_act_fn,
                                                   hidden_layers=[self.hidden_layer_width] * self.half_hidden_layer_depth)

        elif self.algorithm == 'ddqn':

            self.epsilon = kwargs["epsilon"] if ("epsilon" in kwargs) else 0.05
            self.use_pql = self.alg_config["use_pql"] if ("use_pql" in self.alg_config) else False  # Peng's Q(lambda)
            self.cql_alpha = self.alg_config["cql_alpha"] if ("cql_alpha" in self.alg_config) else 0  # Conservative Q-learning (H)
            self.use_cql = True if self.cql_alpha else False
            self.n_steps_return = self.alg_config["n_steps_return"] if ("n_steps_return" in self.alg_config) else False
            self.lambd = self.alg_config["lambd"] if ("lambd" in self.alg_config) else 0.7  # Peng's Q(lambda)

            self.soft_update_target_network = self.alg_config["soft_update_target_network"] if (
                    "soft_update_target_network" in self.alg_config) else False
            self.dueling = self.alg_config["dueling"] if ("dueling" in self.alg_config) else False

            self.alg_type = 'value_based'

            # policy network not needed
            self.f_s2pi0 = nn.ModuleList()  # Placeholder, not used

            # V network not needed
            self.f_s2v = nn.ModuleList()  # Placeholder, not used

            # Q network 1
            self.f_s2q1 = DiscreteActionQNetwork(pre_rl_size, self.action_size, hidden_layers=[self.hidden_layer_width] * self.half_hidden_layer_depth,
                                                 dueling=self.dueling, output_distribution=self.value_distribution,
                                                 act_fn=self.forward_act_fn)

            # Q network 2
            self.f_s2q2 = nn.ModuleList()  # Placeholder, not used

        else:
            raise NotImplementedError("algorithm can only be 'sac' or 'ddqn'")

        self.mse_loss = nn.MSELoss()

        if not self.rnn_state_detach:
            if self.alg_type is 'actor_critic':
                self.optimizer_q = torch.optim.Adam([*self.f_s2q1.parameters(), *self.f_s2q2.parameters(),
                                                     *self.f_s2v.parameters(), *self.encoder_critic.parameters()],
                                                    lr=self.lr)
                self.optimizer_a = torch.optim.Adam([*self.rnn_module.parameters(), *self.f_s2pi0.parameters()],
                                                    lr=self.lr)
            else:
                self.optimizer_q = torch.optim.Adam([*self.f_s2q1.parameters(), *self.f_s2q2.parameters(),
                                                     *self.f_s2v.parameters(), *self.rnn_module.parameters()],
                                                    lr=self.lr)
                self.optimizer_a = None

        else:
            if self.alg_type is 'actor_critic':
                self.optimizer_a = torch.optim.Adam(self.f_s2pi0.parameters(), lr=self.lr)
                self.optimizer_q = torch.optim.Adam([*self.f_s2q1.parameters(), *self.f_s2q2.parameters(),
                                                     *self.f_s2v.parameters(), *self.encoder_critic.parameters()],
                                                    lr=self.lr)
            else:
                self.optimizer_q = torch.optim.Adam([*self.f_s2q1.parameters(), *self.f_s2q2.parameters(),
                                                     *self.f_s2v.parameters()], lr=self.lr)
                self.optimizer_a = None

            self.optimizer_z = torch.optim.Adam(self.rnn_module.parameters(), lr=self.lr)

        if (not self.use_prior_only):
            self.optimizer_b = torch.optim.SGD([self.log_beta], lr=self.lr)

        self.to(device=self.device)

        self.h_t = self.init_hidden_states(1, self.h_forward_size)
        self.a_tm1 = None

        self.zp_tm1 = torch.zeros([1, self.z_deterministic_size + self.z_stochastic_size],
                                  dtype=torch.float32).to(device=self.device)

        if self.weight_init_method is not None:
            if self.weight_init_method == "all_kaiming":
                self.rnn_module.apply(init_weights_kaiming)
            if self.weight_init_method == "z_only_kaiming":
                self.f_hb2muzq.apply(init_weights_kaiming)
                self.f_h2muzp.apply(init_weights_kaiming)

        # target network
        if is_main_network:

            target_net = VLOG(observation_space, full_observation_space, action_space, is_main_network=False, **kwargs)

            # synchronizing target network and main network
            state_dict_tar = target_net.state_dict()
            state_dict = self.state_dict()
            for key in list(target_net.state_dict().keys()):
                state_dict_tar[key] = state_dict[key]
            target_net.load_state_dict(state_dict_tar)

            self.target_net = target_net

    def forward(self, x):
        pass

    def init_states(self):
        self.h_t = self.init_hidden_states(1, self.h_forward_size)
        self.a_tm1 = None

    def select(self, x, action_mask=None, greedy=False, need_other_info=False):

        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x.astype(np.float32).reshape([1, *list(x.shape)])).to(device=self.device)
            x = self.encoder(x)

            if action_mask is not None:
                if isinstance(action_mask, np.ndarray):
                    action_mask = torch.from_numpy(
                        action_mask.astype(np.float32).reshape([1, self.action_size]))

            if self.forward_use_rnn:
                self.h_t, self.zp_tm1, _, _ = self.forward_generative(self.h_t, x, self.a_tm1)
            else:
                e = self.forward_fnn(x)
                if self.z_stochastic_size > 0:
                    muz = self.f_h2muzp(e)
                    logsigz = self.f_h2logsigzp(e)
                    dist = dis.normal.Normal(muz, torch.exp(logsigz))
                    z = dist.sample()

                elif self.z_deterministic_size > 0:
                    z = self.f_h2zp_det(e)

                self.h_t = z

            if self.alg_type == 'actor_critic':

                a = self.f_s2pi0.sample_action(self.h_t, greedy=greedy).reshape([-1])

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

    def init_hidden_states(self, batch_size=1, h_size=256, init_state_generator='zeros', trainable=False):
        if init_state_generator == 'zeros':
            h = torch.zeros((batch_size, h_size), requires_grad=trainable)
        elif init_state_generator == 'normal':
            h = torch.normal(mean=0, std=1, size=(batch_size, h_size), requires_grad=trainable)
        elif init_state_generator == 'uniform':
            h = -1.0 + 2.0 * torch.rand(size=(batch_size, h_size), requires_grad=trainable)
        else:
            raise ValueError("initial state generator must be 'zeros', 'normal' or 'uniform'")

        return h.to(device=self.device)

    def preprocess_data(self, x_seq_batch, x_oracle_seq_batch, action_seq_batch, reward_seq_batch, done_seq_batch,
                        valid_seq_batch, seq_len=8, burn_in_steps=32, mask_seq_batch=None):

        if mask_seq_batch is not None:
            raise NotImplementedError

        start_time = time.time()

        with torch.no_grad():

            batch_size = len(x_seq_batch)
            start_indices = np.zeros(batch_size, dtype=np.int64)
            actual_sampled_len = np.zeros(batch_size, dtype=np.int64)

            x_sampled = np.zeros([batch_size, seq_len + 1, *list(x_seq_batch[0].shape[1:])], dtype=np.float32)
            x_oracle_sampled = np.zeros([batch_size, seq_len + 1, *list(x_oracle_seq_batch[0].shape[1:])],
                                        dtype=np.float32)

            a_sampled = np.zeros([batch_size, seq_len, action_seq_batch[0].shape[-1]], dtype=action_seq_batch[0].dtype)
            r_sampled = np.zeros([batch_size, seq_len], dtype=np.float32)
            d_sampled = np.zeros([batch_size, seq_len], dtype=np.float32)
            v_sampled = np.zeros([batch_size, seq_len], dtype=np.float32)

            actual_burnin_len = np.zeros(batch_size, dtype=np.int64)
            b_post_indices = np.zeros(batch_size, dtype=np.int64)

            max_length_in_batch = 0

            for b in range(batch_size):

                vb = valid_seq_batch[b]
                stps = np.sum(vb).astype(int)

                max_length_in_batch = stps if stps > max_length_in_batch else max_length_in_batch

                start_indices[b] = np.random.randint(- seq_len + int(max(self.n_steps_return, 1)), stps - 1)

                start_index = start_indices[b]

                for tmp, TMP in zip((x_sampled, x_oracle_sampled), (x_seq_batch, x_oracle_seq_batch)):
                    if start_index <= 0 and start_index + seq_len + 1 >= stps + 1:
                        tmp[b, :stps + 1] = TMP[b][:stps + 1]
                        b_post_indices[b] = stps

                    elif start_index <= 0:
                        tmp[b, :(start_index + seq_len + 1)] = TMP[b][:(start_index + seq_len + 1)]
                        b_post_indices[b] = start_index + seq_len

                    elif start_index + seq_len + 1 >= stps + 1:
                        tmp[b, :(stps + 1 - start_index)] = TMP[b][start_index:stps + 1]
                        b_post_indices[b] = stps

                    else:
                        tmp[b] = TMP[b][start_index: (start_index + seq_len + 1)]
                        b_post_indices[b] = start_index + seq_len

                for tmp, TMP in zip((a_sampled, r_sampled, d_sampled, v_sampled),
                                    (action_seq_batch, reward_seq_batch, done_seq_batch, valid_seq_batch)):
                    if start_index <= 0 and start_index + seq_len >= stps:
                        tmp[b, :stps] = TMP[b][:stps]
                        actual_sampled_len[b] = int(stps)

                    elif start_index <= 0:
                        tmp[b, :(start_index + seq_len)] = TMP[b][:(start_index + seq_len)]
                        actual_sampled_len[b] = int(start_index + seq_len)

                    elif start_index + seq_len >= stps:
                        tmp[b, :(stps - start_index)] = TMP[b][start_index:stps]
                        actual_sampled_len[b] = int(stps - start_index)

                    else:
                        tmp[b] = TMP[b][start_index: (start_index + seq_len)]
                        actual_sampled_len[b] = int(seq_len)

            x_sampled = torch.from_numpy(x_sampled.astype(np.float32)).to(device=self.device)
            x_oracle_sampled = torch.from_numpy(x_oracle_sampled.astype(np.float32)).to(device=self.device)
            a_sampled = torch.from_numpy(a_sampled.astype(np.float32)).to(device=self.device)
            r_sampled = torch.from_numpy(r_sampled.astype(np.float32)).to(device=self.device)
            d_sampled = torch.from_numpy(d_sampled.astype(np.float32)).to(device=self.device)
            v_sampled = torch.from_numpy(v_sampled.astype(np.float32)).to(device=self.device)

            x_burnin = torch.zeros([batch_size, burn_in_steps, *list(x_seq_batch[0].shape[1:])],
                                   dtype=torch.float32, requires_grad=False)
            a_burnin = torch.zeros([batch_size, burn_in_steps, *list(action_seq_batch[0].shape[1:])],
                                   dtype=torch.float32, requires_grad=False)
            x_oracle_full = np.zeros([batch_size, max_length_in_batch + 1, *list(x_oracle_seq_batch[0].shape[1:])],
                                     dtype=np.float32)
            for b in range(batch_size):
                x_oracle_full[b, :len(x_oracle_seq_batch[b])] = torch.from_numpy(x_oracle_seq_batch[b].astype(
                    np.float32))

            x_oracle_full = torch.from_numpy(x_oracle_full.astype(np.float32)).to(device=self.device)

            for b in range(batch_size):
                stps = int(np.sum(valid_seq_batch[b]))
                start_index = start_indices[b]

                # forward burn-in samples
                if start_index <= 0:
                    actual_burnin_len[b] = 1
                else:
                    actual_burnin_len[b] = start_index + 1 - max(0, start_index + 1 - burn_in_steps)
                    x_burnin[b, :actual_burnin_len[b]] = torch.from_numpy(
                        x_seq_batch[b][(start_index + 1 - actual_burnin_len[b]): start_index + 1])
                    a_burnin[b, :actual_burnin_len[b]] = torch.from_numpy(
                        action_seq_batch[b][(start_index + 1 - actual_burnin_len[b]): start_index + 1])

            # ------------ backward burn-in ---------------
            x_burnin = x_burnin.to(device=self.device)
            x_oracle_full = x_oracle_full.to(device=self.device)

            if self.image_input:
                phi_burnin = self.encoder(
                    x_burnin.view([batch_size * burn_in_steps, *x_burnin.size()[2:]])).view(
                    [batch_size, burn_in_steps, -1])

                # avoid using too much memory
                phi_oracle_full = torch.zeros(batch_size, max_length_in_batch + 1, self.phi_size_oracle).to(device=self.device)
                for b in range(batch_size):
                    phi_oracle_full[b, :, :] = self.encoder_oracle(x_oracle_full[b, :, :])

            else:
                phi_burnin = self.encoder(x_burnin)

                phi_oracle_full = torch.zeros(batch_size, max_length_in_batch + 1, self.phi_size_oracle).to(device=self.device)
                for b in range(batch_size):
                    phi_oracle_full[b, :, :] = self.encoder_oracle(x_oracle_full[b, :, :])

            if self.oracle_use_rnn:
                b = self.init_hidden_states(batch_size, self.h_oracle_size)

                b_tensor_flipped, _ = self.rnn_oracle(torch.flip(phi_oracle_full, dims=[1]),
                                                        b.view(1, b.size()[0], b.size()[-1]))

                b_tensor = torch.flip(b_tensor_flipped, dims=[1])
                b_post_sampled = b_tensor[np.arange(0, batch_size), b_post_indices, :].detach()

                # target network

                if self.target_net_include_rnn:
                    b_tar = self.init_hidden_states(batch_size, self.h_oracle_size)
                    b_tensor_tar_flipped, _ = self.target_net.rnn_oracle(torch.flip(phi_oracle_full, dims=[1]),
                                                                           b_tar.view(1, b_tar.size()[0],
                                                                                      b_tar.size()[-1]))
                    b_tensor_tar = torch.flip(b_tensor_tar_flipped, dims=[1])
                    b_post_sampled_tar = b_tensor_tar[np.arange(0, batch_size), b_post_indices, :].detach()
                else:
                    b_tar = b.detach().clone()
                    b_tensor_tar = b_tensor.detach().clone()
                    b_post_sampled_tar = b_post_sampled.detach().clone()

            else:
                b_tensor = self.oracle_fnn(phi_oracle_full)
                if self.target_net_include_rnn:
                    b_tensor_tar = self.target_net.oracle_fnn(phi_oracle_full)
                else:
                    b_tensor_tar = b_tensor.detach().clone()

                b_post_sampled = None
                b_post_sampled_tar = None

            # -------------- forward burn-in ------------------
            h_series = []
            h_tar_series = []

            h = self.init_hidden_states(batch_size, self.h_forward_size)
            h_tar = self.init_hidden_states(batch_size, self.h_forward_size)

            h, _, _, _ = self.forward_generative(h, phi_burnin[:, 0], None)
            h_tar, _, _, _ = self.forward_generative(h_tar, phi_burnin[:, 0], None)

            h_series.append(h)  # first step may be sampled later
            h_tar_series.append(h_tar)

            for t in range(1, np.max(actual_burnin_len)):  # not necessary + 1

                if self.use_prior_only:
                    h, _, _, _ = self.forward_generative(h, phi_burnin[:, t], a_burnin[:, t - 1])
                    if self.target_net_include_rnn:
                        h_tar, _, _, _ = self.target_net.forward_generative(h_tar, phi_burnin[:, t], a_burnin[:, t - 1])
                else:

                    b4h_indices = np.maximum(0, start_indices + t - actual_burnin_len).astype(np.int64)

                    b4h_t = b_tensor[np.arange(0, batch_size), b4h_indices, :]
                    h, _, _, _ = self.forward_inference(h, phi_burnin[:, t], b4h_t, a_burnin[:, t - 1])

                    if self.target_net_include_rnn:
                        b4h_t_tar = b_tensor_tar[np.arange(0, batch_size), b4h_indices, :]
                        h_tar, _, _, _ = self.target_net.forward_inference(h_tar, phi_burnin[:, t], b4h_t_tar,
                                                                           a_burnin[:, t - 1])

                h_series.append(h)

                if self.target_net_include_rnn:
                    h_tar_series.append(h_tar)
                else:
                    h_tar_series.append(h.detach())

            h_tensor = torch.stack(h_series, dim=1)
            h_tensor_tar = torch.stack(h_tar_series, dim=1)

            h_pre_sampled = h_tensor[np.arange(0, batch_size), actual_burnin_len - 1, :].detach()
            h_pre_sampled_tar = h_tensor_tar[np.arange(0, batch_size), actual_burnin_len - 1, :].detach()

            # print("forward_burnin time:", time.time() - start_time)

            if self.target_net_include_rnn:
                h_tar_series_sampled, h_tp1_tar_series_sampled, hp_tar_series_sampled, _, _, _, _, _, _ = \
                    self.target_net.get_series_data_with_rnn(x_sampled, x_oracle_sampled, a_sampled, r_sampled,
                                                             d_sampled, v_sampled, h_pre_sampled_tar,
                                                             b_post_sampled_tar, batch_size, seq_len)

        # with grad now
        h_series_sampled, h_tp1_series_sampled, hp_series_sampled, zp_series_sampled, muzp_series_sampled, logsigzp_series_sampled,\
        zq_series_sampled, muzq_series_sampled, logsigzq_series_sampled = \
            self.get_series_data_with_rnn(x_sampled, x_oracle_sampled, a_sampled, r_sampled, d_sampled, v_sampled,
                                          h_pre_sampled, b_post_sampled, batch_size, seq_len)

        a_sampled = a_sampled[:, 1:]
        r_sampled = r_sampled[:, 1:]
        d_sampled = d_sampled[:, 1:]
        v_sampled = v_sampled[:, 1:]

        if not self.target_net_include_rnn:
            h_tar_series_sampled = [h.detach() for h in h_series_sampled]
            h_tp1_tar_series_sampled = [h.detach() for h in h_tp1_series_sampled]

        # print("get_series_data time:", time.time() - start_time)

        return h_series_sampled, h_tp1_series_sampled, h_tar_series_sampled, h_tp1_tar_series_sampled, hp_series_sampled,\
               zp_series_sampled, muzp_series_sampled, logsigzp_series_sampled,\
               zq_series_sampled, muzq_series_sampled, logsigzq_series_sampled,\
               x_sampled[:, 1:-1], x_sampled[:, 2:], x_oracle_sampled[:, 1:-1], x_oracle_sampled[:, 2:],\
               a_sampled, r_sampled, d_sampled, v_sampled

    def preprocess_data_fnn(self, x_seq_batch, x_oracle_seq_batch, action_seq_batch, reward_seq_batch, done_seq_batch,
                            valid_seq_batch, seq_len=8, mask_seq_batch=None):
        with torch.no_grad():

            batch_size = len(x_seq_batch)
            start_indices = np.zeros(batch_size, dtype=np.int64)

            x_sampled = np.zeros([batch_size, seq_len + 1, *list(x_seq_batch[0].shape[1:])], dtype=np.float32)
            x_oracle_sampled = np.zeros([batch_size, seq_len + 1, *list(x_oracle_seq_batch[0].shape[1:])],
                                        dtype=np.float32)

            if len(action_seq_batch[0].shape) > 1:
                action_dim = action_seq_batch[0].shape[-1]
                a_sampled = np.zeros([batch_size, seq_len, action_dim], dtype=action_seq_batch[0].dtype)
            else:
                action_dim = 1
                a_sampled = np.zeros([batch_size, seq_len], dtype=action_seq_batch[0].dtype)

            r_sampled = np.zeros([batch_size, seq_len], dtype=np.float32)
            d_sampled = np.zeros([batch_size, seq_len], dtype=np.float32)
            v_sampled = np.zeros([batch_size, seq_len], dtype=np.float32)
            if mask_seq_batch is None:
                m_sampled = None
            else:
                m_sampled = np.ones([batch_size, seq_len, self.action_size])

            for b in range(batch_size):
                vb = valid_seq_batch[b]
                stps = np.sum(vb).astype(int)

                start_indices[b] = np.random.randint(- seq_len + int(max(self.n_steps_return, 1)), stps - 1)
                start_index = start_indices[b]

                for tmp, TMP in zip((x_sampled, x_oracle_sampled), (x_seq_batch, x_oracle_seq_batch)):
                    if start_index <= 0 and start_index + seq_len + 1 >= stps + 1:
                        tmp[b, :stps + 1] = TMP[b][:stps + 1]
                    elif start_index <= 0:
                        tmp[b, :(start_index + seq_len + 1)] = TMP[b][:(start_index + seq_len + 1)]
                    elif start_index + seq_len + 1 >= stps + 1:
                        tmp[b, :(stps + 1 - start_index)] = TMP[b][start_index:stps + 1]
                    else:
                        tmp[b] = TMP[b][start_index: (start_index + seq_len + 1)]

                if mask_seq_batch is not None:
                    zips = zip((a_sampled, r_sampled, d_sampled, v_sampled, m_sampled),
                               (action_seq_batch, reward_seq_batch, done_seq_batch, valid_seq_batch, mask_seq_batch))
                else:
                    zips = zip((a_sampled, r_sampled, d_sampled, v_sampled),
                               (action_seq_batch, reward_seq_batch, done_seq_batch, valid_seq_batch))

                for tmp, TMP in zips:
                    if start_index <= 0 and start_index + seq_len >= stps:
                        tmp[b, :stps] = TMP[b][:stps]
                    elif start_index <= 0:
                        tmp[b, :(start_index + seq_len)] = TMP[b][:(start_index + seq_len)]
                    elif start_index + seq_len >= stps:
                        tmp[b, :(stps - start_index)] = TMP[b][start_index:stps]
                    else:
                        tmp[b] = TMP[b][start_index: (start_index + seq_len)]

            x_sampled_tensor = torch.from_numpy(x_sampled[:, :-1].astype(np.float32)).to(device=self.device)
            xp_sampled_tensor = torch.from_numpy(x_sampled[:, 1:].astype(np.float32)).to(device=self.device)
            x_oracle_sampled_tensor = torch.from_numpy(x_oracle_sampled[:, :-1].astype(np.float32)).to(device=self.device)
            xp_oracle_sampled_tensor = torch.from_numpy(x_oracle_sampled[:, 1:].astype(np.float32)).to(device=self.device)
            a_sampled = a_sampled.reshape([a_sampled.shape[0], a_sampled.shape[1], action_dim])
            a_sampled_tensor = torch.from_numpy(a_sampled.astype(np.float32)).to(device=self.device)
            r_sampled_tensor = torch.from_numpy(r_sampled.astype(np.float32)).to(device=self.device)
            d_sampled_tensor = torch.from_numpy(d_sampled.astype(np.float32)).to(device=self.device)
            v_sampled_tensor = torch.from_numpy(v_sampled.astype(np.float32)).to(device=self.device)

            if mask_seq_batch is None:
                m_sampled_tensor = None
            else:
                m_sampled_tensor = torch.from_numpy(m_sampled.astype(np.float32)).to(device=self.device)

        return x_sampled_tensor, xp_sampled_tensor, x_oracle_sampled_tensor, xp_oracle_sampled_tensor, \
               a_sampled_tensor, r_sampled_tensor, d_sampled_tensor, v_sampled_tensor, m_sampled_tensor

    def forward_generative(self, h_tm1, x_t, a_tm1, reparameterization=True):

        if self.z_deterministic_size == 0 and self.z_stochastic_size == 0:
            rnn_input = x_t
            zp_tm1 = torch.tensor([0])
            muzp_tm1 = torch.tensor([0])
            logsigzp_tm1 = torch.tensor([0])

        elif a_tm1 is not None:
            if self.action_to_zp and self.alg_type == 'value_based':
                a_tm1_one_hot = F.one_hot(a_tm1.view([-1]).to(torch.int64), num_classes=self.action_size).to(
                    torch.float32).to(device=self.device)
                pre_zp_input = torch.cat([h_tm1, a_tm1_one_hot], dim=-1)
            elif self.action_to_zp and self.alg_type == 'actor_critic':
                pre_zp_input = torch.cat([h_tm1, a_tm1.view([h_tm1.size()[0], -1])], dim=-1)
            else:
                pre_zp_input = h_tm1

            if self.z_stochastic_size == 0 and self.z_deterministic_size > 0:
                zp_tm1 = self.f_h2zp_det(pre_zp_input)
                rnn_input = torch.cat((x_t, zp_tm1), dim=-1)
                muzp_tm1 = torch.tensor([0])
                logsigzp_tm1 = torch.tensor([0])

            elif self.z_deterministic_size == 0 and self.z_stochastic_size > 0:
                muzp_tm1 = self.f_h2muzp(pre_zp_input)
                logsigzp_tm1 = self.f_h2logsigzp(pre_zp_input)
                m = torch.distributions.normal.Normal(muzp_tm1, torch.exp(logsigzp_tm1))
                zp_tm1 = m.rsample() if reparameterization else m.sample()
                rnn_input = torch.cat((x_t, zp_tm1), dim=-1)
            else:
                raise NotImplementedError

        else:  # (a_tm1 is None) first step zp unit Gaussian
            muzp_tm1 = torch.zeros([h_tm1.size()[0], self.z_size],
                                   requires_grad=False).to(device=self.device)
            logsigzp_tm1 = torch.zeros_like(muzp_tm1, requires_grad=False).to(device=self.device)
            sigzp_tm1 = torch.ones_like(muzp_tm1, requires_grad=False).to(device=self.device)
            m = torch.distributions.normal.Normal(muzp_tm1, sigzp_tm1)
            zp_tm1 = m.sample().detach_()
            rnn_input = torch.cat((x_t, zp_tm1), dim=-1)

        h_t = self.rnn_forward(rnn_input, h_tm1)

        return h_t, zp_tm1, muzp_tm1, logsigzp_tm1

    def forward_inference(self, h_tm1, x_t, b_tm1, a_tm1, reparameterization=True):
        muzq_tm1 = None
        logsigzq_tm1 = None

        if not self.action_to_zq:
            pre_zq_input = torch.cat((h_tm1, b_tm1), dim=-1)
        else:
            a_tm1_one_hot = F.one_hot(a_tm1.view([-1]).to(torch.int64), num_classes=self.action_size).to(
                torch.float32).to(device=self.device)
            pre_zq_input = torch.cat((h_tm1, b_tm1, a_tm1_one_hot), dim=-1)

        if self.z_stochastic_size == 0 and self.z_deterministic_size > 0:
            zq_tm1 = self.f_hb2zq_det(pre_zq_input)

        elif self.z_deterministic_size == 0 and self.z_stochastic_size > 0:
            muzq_tm1 = self.f_hb2muzq(pre_zq_input)
            logsigzq_tm1 = self.f_hb2logsigzq(pre_zq_input)
            m = torch.distributions.normal.Normal(muzq_tm1, torch.exp(logsigzq_tm1))
            zq_tm1 = m.rsample() if reparameterization else m.sample()
        else:
            raise NotImplementedError

        rnn_input = torch.cat((x_t, zq_tm1), dim=-1)
        h_t = self.rnn_forward(rnn_input, h_tm1)

        return h_t, zq_tm1, muzq_tm1, logsigzq_tm1

    def get_series_data_with_rnn(self, x, x_oracle, a, r, d, v, h_pre, b_post, batch_size, seq_len):

        # ------------ compute backward rnn --------
        if b_post is None:
            b_post = self.init_hidden_states(batch_size, self.h_oracle_size)

        b_post = b_post.detach()

        x_oracle = x_oracle.detach_()[:, :-1, :].clone()  # final step already burn-in'ed

        if self.image_input:
            x = self.encoder(x.view([batch_size * (seq_len + 1), *x.size()[2:]])).view([batch_size, (seq_len + 1), -1])
            x_oracle = self.encoder_oracle(x_oracle.view(
                [batch_size * seq_len, *x_oracle.size()[2:]])).view([batch_size, seq_len, -1])
        else:
            x = self.encoder(x)
            x_oracle = self.encoder_oracle(x_oracle)

        if self.oracle_use_rnn:
            b_tensor, _ = self.rnn_oracle(torch.flip(x_oracle, dims=[1]), b_post.view(
                1, b_post.size()[0], b_post.size()[-1]))
        else:
            b_tensor = self.oracle_fnn(x_oracle)

        # ------------ compute forward rnn --------
        if h_pre is None:
            h = self.init_hidden_states(batch_size, self.h_forward_size)
        else:
            h = h_pre

        zp_series = []
        zq_series = []
        h_series = []
        hp_series = []

        muzp_series = []
        logsigzp_series = []
        muzq_series = []
        logsigzq_series = []

        h_tp1_series = []

        for stp in range(1, seq_len + 1):
            if not self.use_prior_only:
                hp, zp_tm1, muzp_tm1, logsigzp_tm1 = self.forward_generative(h, x[:, stp], a[:, stp - 1])

                h, zq_tm1, muzq_tm1, logsigzq_tm1 = self.forward_inference(h, x[:, stp],
                                                                           b_tensor[:, seq_len - stp], a[:, stp - 1])

            else:
                h, zp_tm1, muzp_tm1, logsigzp_tm1 = self.forward_generative(h, x[:, stp], a[:, stp - 1])
                hp = h

            if stp < seq_len:
                h_series.append(h)
                hp_series.append(hp)

            if stp > 1:
                h_tp1_series.append(h)

                if not self.use_prior_only:
                    zq_series.append(zq_tm1)
                zp_series.append(zp_tm1)

                if self.z_deterministic_size == 0 and self.z_stochastic_size > 0:
                    if not self.use_prior_only:
                        muzq_series.append(muzq_tm1)
                        logsigzq_series.append(logsigzq_tm1)

                    muzp_series.append(muzp_tm1)
                    logsigzp_series.append(logsigzp_tm1)

        return h_series, h_tp1_series, hp_series, zp_series, muzp_series, logsigzp_series, zq_series, muzq_series, logsigzq_series

    def pretrain_oracle(self, X, X_oracle, A, R, D, V, L, seq_len=8, batch_size=8, burn_in_steps=32, times=1000):

        beta_original = self.beta

        # if self.z_stochastic_size > 0 and self.z_deterministic_size == 0:
        #     for param in self.f_h2muzp.parameters():
        #         param.requires_grad = False
        #
        #     for param in self.f_h2logsigzp.parameters():
        #         param.requires_grad = False
        #
        # elif self.z_stochastic_size == 0 and self.z_deterministic_size > 0:
        #     for param in self.f_h2zp_det.parameters():
        #         param.requires_grad = False
        # else:
        #     return 0

        self.beta = 0
        for _ in range(times):
            loss = self.learn(X, X_oracle, A, R, D, V, L, seq_len, batch_size, burn_in_steps)

        self.beta = beta_original

        # if self.z_stochastic_size > 0 and self.z_deterministic_size == 0:
        #     for param in self.f_h2muzp.parameters():
        #         param.requires_grad = True
        #
        #     for param in self.f_h2logsigzp.parameters():
        #         param.requires_grad = True
        #
        # elif self.z_stochastic_size == 0 and self.z_deterministic_size > 0:
        #     for param in self.f_h2zp_det.parameters():
        #         param.requires_grad = True

        return loss

    def learn(self, X, X_oracle, A, R, D, V, L, seq_len=8, batch_size=8, burn_in_steps=32, action_masks=None):

        # L is length of each episode (if we have s_0, s_1, ...., s_T in epsiode e, then L[e] = T]
        L = np.array(L).astype(np.float32)
        weights = L + 2 * seq_len - 2  # see preprocess_data
        e_samples = np.random.choice(len(L), batch_size, p=weights / weights.sum())  # sample with equal p for each step

        # print(e_samples)
        x_seq_batch = [X[e_samples[i]] for i in range(batch_size)]
        x_oracle_seq_batch = [X_oracle[e_samples[i]] for i in range(batch_size)]
        action_seq_batch = [A[e_samples[i]] for i in range(batch_size)]
        reward_seq_batch = [R[e_samples[i]] for i in range(batch_size)]
        done_seq_batch = [D[e_samples[i]] for i in range(batch_size)]

        if V is not None:
            valid_seq_batch = [V[e_samples[i]] for i in range(batch_size)]
        else:
            valid_seq_batch = [np.ones_like(D[e_samples[i]]) for i in range(batch_size)]

        if action_masks is not None:
            mask_seq_batch = [action_masks[e_samples[i]] for i in range(batch_size)]
        else:
            mask_seq_batch = None

        # ------------ pre-process numpy data ------------

        start_time = time.time()

        if self.forward_use_rnn:
            h_series, h_tp1_series, h_tar_series, h_tp1_tar_series, hp_series, \
                zp_series, muzp_series, logsigzp_series, \
                zq_series, muzq_series, logsigzq_series, \
                x, xp, x_oracle, xp_oracle, a, r, d, v = self.preprocess_data(
                x_seq_batch, x_oracle_seq_batch, action_seq_batch, reward_seq_batch, done_seq_batch, valid_seq_batch,
                seq_len + 1, burn_in_steps=burn_in_steps, mask_seq_batch=mask_seq_batch)

            h_tensor = torch.stack(h_series, dim=1)
            hp_tensor = torch.stack(hp_series, dim=1)
            if self.rnn_state_detach:
                h_tensor = h_tensor.detach()
                hp_tensor = hp_tensor.detach()

            h_tar_tensor = torch.stack(h_tar_series, dim=1).detach()

            h_tp1_tensor = torch.stack(h_tp1_series, dim=1).detach()  # only for computing rl target
            h_tp1_tar_tensor = torch.stack(h_tp1_tar_series, dim=1).detach()

            # if self.update_times < 10:
            #     print("preprocess time:", time.time() - start_time)

            if self.z_stochastic_size == 0 and self.z_deterministic_size > 0:
                zp_tensor = torch.stack(zp_series, dim=1)
                if not self.use_prior_only:
                    zq_tensor = torch.stack(zq_series, dim=1)
            elif self.z_deterministic_size == 0 and self.z_stochastic_size > 0:

                muzp_tensor = torch.stack(muzp_series, dim=1)
                logsigzp_tensor = torch.stack(logsigzp_series, dim=1)

                if not self.use_prior_only:
                    muzq_tensor = torch.stack(muzq_series, dim=1)
                    logsigzq_tensor = torch.stack(logsigzq_series, dim=1)

        else:
            x, xp, x_oracle, xp_oracle, a, r, d, v, m = \
                self.preprocess_data_fnn(x_seq_batch, x_oracle_seq_batch, action_seq_batch, reward_seq_batch,
                                         done_seq_batch, valid_seq_batch, seq_len, mask_seq_batch=mask_seq_batch)  # Here is seq_len, no need +1
            # print(x.shape)
            phi = self.encoder(x.view([batch_size * seq_len, *x.size()[2:]])).view([batch_size, seq_len, -1])
            phi_oracle = self.encoder_oracle(x_oracle.view([batch_size * seq_len, *x_oracle.size()[2:]])).view(
                [batch_size, seq_len, -1])

            phi_tp1 = self.encoder(xp.view([batch_size * seq_len, *xp.size()[2:]])).view([batch_size, seq_len, -1])
            phi_oracle_tp1 = self.encoder_oracle(xp_oracle.view([batch_size * seq_len, *xp_oracle.size()[2:]])).view(
                [batch_size, seq_len, -1])

            # phi_tar = self.target_net.encoder(x.view([batch_size * seq_len, *x.size()[2:]])).view(
            #     [batch_size, seq_len, -1])
            phi_tp1_tar = self.target_net.encoder(
                xp.view([batch_size * seq_len, *xp.size()[2:]])).view([batch_size, seq_len, -1])

            # phi_oracle_tar = self.target_net.encoder(x_oracle.view([batch_size * seq_len, *x_oracle.size()[2:]])).view(
            #     [batch_size, seq_len, -1])
            phi_oracle_tp1_tar = self.target_net.encoder_oracle(
                xp_oracle.view([batch_size * seq_len, *xp_oracle.size()[2:]])).view([batch_size, seq_len, -1])

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

                    with torch.no_grad():
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
                    zp_tensor = self.f_h2zp_det(self.forward_fnn(phi))
                    zq_tensor = self.f_hb2zq_det(self.oracle_fnn(phi_oracle))

                    h_tensor = zq_tensor
                    hp_tensor = zp_tensor

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

                    e_tp1 = self.forward_fnn(phi_tp1).detach()
                    h_tp1_tensor = self.f_h2zp_det(e_tp1).detach()

                    e_tp1_tar = self.target_net.forward_fnn(phi_tp1_tar).detach()
                    h_tp1_tar_tensor = self.target_net.f_h2zp_det(e_tp1_tar).detach()
                else:
                    raise NotImplementedError

        if self.alg_type is 'actor_critic':
            phi_critic = self.encoder_critic(x_oracle)
            phi_critic_tar = self.target_net.encoder_critic(x_oracle).detach()
            phi_tp1_critic = self.encoder_critic(xp_oracle).detach()
            phi_tp1_critic_tar = self.target_net.encoder_critic(xp_oracle).detach()

        # ------------ compute divergence between z^p and z^q -------------
        if not self.use_prior_only:
            if self.z_stochastic_size == 0 and self.z_deterministic_size > 0:

                loss_z = 1. / self.action_size * torch.mean(
                    torch.sum(0.5 * (zq_tensor - zp_tensor).pow(2), dim=-1) * v)  # deterministic z only

                if self.verbose and np.random.rand() < 0.005 * self.verbose:
                    tmp = np.array2string(zp_tensor[0, 0, :6].detach().cpu().numpy(), precision=4, separator=',  ')
                    print("zp = " + tmp)
                    tmp = np.array2string(zq_tensor[0, 0, :6].detach().cpu().numpy(), precision=4, separator=',  ')
                    print("zq = " + tmp)

            elif self.z_deterministic_size == 0 and self.z_stochastic_size > 0:

                if self.verbose and np.random.rand() < 0.005 * self.verbose:
                    tmp = np.array2string(muzp_tensor[0, 0, :6].detach().cpu().numpy(), precision=4, separator=',  ')
                    print("muzp = " + tmp)
                    tmp = np.array2string(muzq_tensor[0, 0, :6].detach().cpu().numpy(), precision=4, separator=',  ')
                    print("muzq = " + tmp)

                    tmp = np.array2string(logsigzp_tensor[0, 0, :6].detach().cpu().exp().numpy(),
                                          precision=4, separator=', ')
                    print("sigzp = " + tmp)
                    tmp = np.array2string(logsigzq_tensor[0, 0, :6].detach().cpu().exp().numpy(),
                                          precision=4, separator=', ')
                    print("sigzq = " + tmp)

                if not self.mmd:
                    loss_z = torch.mean(
                        torch.sum(logsigzp_tensor - logsigzq_tensor + ((muzp_tensor - muzq_tensor).pow(2) + torch.exp(
                            logsigzq_tensor * 2)) / (2.0 * torch.exp(logsigzp_tensor * 2)) - 0.5, dim=-1) * v)
                else:
                    loss_z = compute_mmd(zp_tensor, zq_tensor, v)

                loss_z = 1. / self.action_size * loss_z
            else:
                raise NotImplementedError
        else:
            loss_z = torch.tensor(0)

        # ------------ compute (posterior) value prediction loss  -------------
        if self.algorithm == 'ddqn':

            gamma = self.gamma

            # q_tensor = self.f_s2q1(h_tensor)
            qp_tensor = self.f_s2q1(h_tp1_tensor).detach()

            qp_tensor_tar = self.target_net.f_s2q1(h_tp1_tar_tensor).detach()

            if m is not None:
                mp = torch.ones_like(m)
                mp[:, :-1] = m[:, 1:]
                qp_tensor = qp_tensor * mp - (1 - mp) * INFINITY
                qp_tensor_tar = qp_tensor_tar * mp - (1 - mp) * INFINITY

            a_one_hot = F.one_hot(a[:, :, 0].to(torch.int64), num_classes=self.action_size).to(torch.float32)

            if self.n_steps_return:
                # n-steps double Q-learning
                r_padded = torch.cat(
                    (r, torch.zeros([batch_size, self.n_steps_return - 1], dtype=torch.float32, device=self.device)),
                    dim=1)

                qp_tensor_tar_padded = torch.cat((qp_tensor_tar, torch.zeros(
                    [batch_size, self.n_steps_return - 1, self.action_size], dtype=torch.float32, device=self.device)),
                                                 dim=1)

                # pad done
                d_padded = torch.cat(
                    (d, torch.zeros([batch_size, self.n_steps_return - 1], dtype=torch.float32, device=self.device)),
                    dim=1)
                v_padded = torch.cat(
                    (v, torch.zeros([batch_size, self.n_steps_return - 1], dtype=torch.float32, device=self.device)),
                    dim=1)
                has_done = torch.sum(d, dim=-1).detach().cpu().numpy()
                assert np.max(has_done) <= 1
                for batch_id in range(has_done.shape[0]):
                    if has_done[batch_id]:
                        d_ind = torch.argmax(d[batch_id])
                        d_padded[batch_id, d_ind:] = 1
                        v_padded[batch_id, d_ind: d_ind + self.n_steps_return] = 1

                a_greedy = torch.argmax(qp_tensor, dim=-1, keepdim=False)

                # greedy action is selected using the main network
                a_greedy_one_hot = F.one_hot(a_greedy.to(torch.int64), num_classes=self.action_size).to(torch.float32)
                a_greedy_one_hot_padded = torch.cat(
                    (a_greedy_one_hot, torch.zeros([batch_size, self.n_steps_return - 1, self.action_size],
                                                   dtype=torch.float32, device=self.device)), dim=1)

                qp_tensor_tar_greedy = (qp_tensor_tar_padded * a_greedy_one_hot_padded).sum(dim=-1)

                q_target = torch.zeros([batch_size, seq_len], dtype=torch.float32, device=self.device).detach()

                # for t in range(seq_len):
                for tt in range(self.n_steps_return):
                    q_target = q_target + self.gamma ** tt * r_padded[:, tt: tt + seq_len]

                q_target = q_target + (1 - d_padded[:, self.n_steps_return - 1:]) * (
                        self.gamma ** self.n_steps_return) * qp_tensor_tar_greedy[:, self.n_steps_return - 1:]

                v = v_padded[:, self.n_steps_return - 1:]  # mask for n-steps return

            elif self.use_pql:
                a_greedy_tp1 = torch.argmax(qp_tensor[:, -1], dim=-1, keepdim=False)
                a_greedy_one_hot_tp1 = F.one_hot(a_greedy_tp1.to(torch.int64), num_classes=self.action_size).to(
                    torch.float32)
                q_target_tp1 = torch.sum(qp_tensor_tar[:, -1] * a_greedy_one_hot_tp1, dim=-1)

                q_target = torch.zeros(batch_size, seq_len, dtype=torch.float32).to(device=self.device)

                for t in reversed(range(seq_len)):
                    a_greedy_tp1 = torch.argmax(qp_tensor[:, t], dim=-1, keepdim=False)
                    a_greedy_one_hot_tp1 = F.one_hot(a_greedy_tp1.to(torch.int64), num_classes=self.action_size).to(
                        torch.float32)
                    q_max_tp1 = torch.sum(qp_tensor_tar[:, t] * a_greedy_one_hot_tp1, dim=-1)
                    q_target[:, t] = r[:, t] + (1 - d[:, t]) * self.gamma * (q_max_tp1 + self.lambd * (
                            q_target_tp1 - q_max_tp1))
                    q_target_tp1 = q_target[:, t]

                q_target = q_target.detach()

            else:
                # double Q-learning
                a_greedy = torch.argmax(qp_tensor, dim=-1, keepdim=False)
                # greedy action is selected using the main network
                a_greedy_one_hot = F.one_hot(a_greedy.to(torch.int64), num_classes=self.action_size).to(torch.float32)
                q_target = (r + (1 - d) * gamma * torch.sum(qp_tensor_tar.detach() * a_greedy_one_hot, dim=-1))

            q_target_expand = q_target.view([*q_target.size(), 1]).repeat_interleave(self.action_size, dim=-1)

            if self.tabular_like_reg:
                hbs = int(batch_size / 2)
                loss_critic = torch.mean(torch.sum(
                    - self.f_s2q1.get_log_prob(h_tensor[:hbs], q_target_expand[:hbs]) * a_one_hot[:hbs], dim=-1) * v[:hbs])

                loss_critic = loss_critic + self.tabular_like_reg * torch.mean(torch.sum(
                    - self.f_s2q1.get_log_prob(h_tp1_tensor[hbs:], qp_tensor_tar[hbs:]) * a_one_hot[hbs:], dim=-1) * v[hbs:])

            else:
                loss_critic = torch.mean(
                    torch.sum(- self.f_s2q1.get_log_prob(h_tensor, q_target_expand) * a_one_hot, dim=-1) * v)

            if self.use_cql:
                loss_cql = self.cql_alpha * torch.mean(torch.logsumexp(self.f_s2q1(h_tensor), dim=-1)
                                                       - torch.sum(a_one_hot * self.f_s2q1(h_tensor), dim=-1))
            else:
                loss_cql = 0

            if (not self.use_prior_only) and (self.pq_div_threshold >= 0) and torch.sum(v).item() > 0:
                loss_critic_prior = torch.mean(
                    torch.sum(- self.f_s2q1.get_log_prob(hp_tensor, q_target_expand) * a_one_hot, dim=-1) * v)

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

            if self.rnn_state_detach:
                self.optimizer_z.zero_grad()
            self.optimizer_q.zero_grad()

            if not self.use_prior_only:
                (torch.exp(self.log_beta.detach()) * loss_z + loss_critic + loss_cql).backward()  # 2 * loss_z because value and policy both have KLD loss
            else:
                (loss_critic + loss_cql).backward()

            if self.rnn_state_detach:
                self.optimizer_z.step()  # parameters of z included in a, q if rnn_state_detach = False
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

            elif (not self.use_prior_only) and (self.pq_div_threshold >= 1) and (self.alpha == 0) and torch.sum(v).item() > 0:

                self.recent_loss_set[self.update_times % self.recent_loss_set_size] = loss_critic.detach().cpu().item()
                self.recent_loss_prior_set[self.update_times % self.recent_loss_set_size] = loss_critic_prior.detach().cpu().item()

                fluc_loss = np.std(self.recent_loss_set)
                fluc_loss_prior = np.std(self.recent_loss_prior_set)

                if self.update_times >= self.recent_loss_set_size:
                    if fluc_loss_prior > self.pq_div_threshold * fluc_loss:
                        logbeta_grad = 1
                    elif fluc_loss_prior < fluc_loss / self.pq_div_threshold:
                        logbeta_grad = -1
                    else:
                        logbeta_grad = 0

                    loss_beta = - torch.mean(self.log_beta * logbeta_grad)

                    if self.verbose and np.random.rand() < 0.005 * self.verbose:
                        print("log beta = {}, fluc_loss_q = {}, fluc_loss_q_prior= {}".format(
                            self.log_beta.detach().item(), fluc_loss, fluc_loss_prior))

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

                if self.pq_div_threshold == 0:
                    loss_beta = - torch.mean(self.log_beta * torch.sign(loss_diff) * (torch.abs(torch.atanh(loss_diff)) ** self.atanh_power))
                else:
                    if loss_diff < - (self.pq_div_threshold - 1) / (self.pq_div_threshold + 1):
                        logbeta_grad = -1
                    elif loss_diff > (self.pq_div_threshold - 1) / (self.pq_div_threshold + 1):
                        logbeta_grad = 1
                    else:
                        logbeta_grad = 0
                    loss_beta = - torch.mean(self.log_beta * logbeta_grad)

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

        elif self.algorithm == "sac":

            # --------- critic loss -----------
            EPS = 1e-6  # Avoid NaN
            REG = 1e-3

            v = v.view([*v.size(), 1]).detach()
            r = r.view([*r.size(), 1]).detach()
            d = d.view([*d.size(), 1]).detach()
            a = a.detach()

            if isinstance(self.beta_h, str):
                beta_h = torch.exp(self.log_beta_h).detach()
            else:
                beta_h = np.float32(self.beta_h)

            mua_tensor, logsiga_tensor = self.f_s2pi0(h_tensor)
            siga_tensor = torch.exp(logsiga_tensor)

            mu_prob = dis.normal.Normal(mua_tensor, siga_tensor)

            sampled_u = mu_prob.sample()
            sampled_a = torch.tanh(sampled_u)
            log_pi_exp = torch.sum(mu_prob.log_prob(sampled_u).clamp(-20, 10), dim=-1,
                                   keepdim=True) - torch.sum(
                torch.log(1 - sampled_a.pow(2) + EPS), dim=-1, keepdim=True)

            v_tensor = self.f_s2v(phi_critic)
            vp_tensor = self.target_net.f_s2v(phi_tp1_critic_tar)

            q_tensor_1 = self.f_s2q1(phi_critic, a)
            q_tensor_2 = self.f_s2q2(phi_critic, a)

            sampled_q = torch.min(self.f_s2q1(phi_critic, sampled_a).detach(), self.f_s2q2(phi_critic, sampled_a).detach())

            q_exp = sampled_q

            v_tar = (q_exp - beta_h * log_pi_exp.detach()).detach()
            loss_v = 0.5 * self.mse_loss(v_tensor * v, v_tar * v)

            if self.tabular_like_reg:
                hbs = int(batch_size / 2)

                q_tensor_tar_1 = self.target_net.f_s2q1(phi_critic_tar[hbs:], a[hbs:]).detach()
                q_tensor_tar_2 = self.target_net.f_s2q2(phi_critic_tar[hbs:], a[hbs:]).detach()

                loss_q = 0.5 * self.mse_loss(q_tensor_1[:hbs] * v[:hbs],
                                             (r[:hbs] + (1 - d[:hbs]) * self.gamma * vp_tensor[:hbs].detach()) * v[:hbs]) \
                         + 0.5 * self.mse_loss(q_tensor_2[:hbs] * v[:hbs],
                                               (r[:hbs] + (1 - d[:hbs]) * self.gamma * vp_tensor[:hbs].detach()) * v[:hbs])
                loss_q = loss_q + self.tabular_like_reg * (0.5 * self.mse_loss(q_tensor_1[hbs:] * v[hbs:], q_tensor_tar_1 * v[hbs:])
                         + 0.5 * self.mse_loss(q_tensor_2[hbs:] * v[hbs:], q_tensor_tar_2 * v[hbs:]))
            else:
                loss_q = 0.5 * self.mse_loss(q_tensor_1 * v, (r + (1 - d) * self.gamma * vp_tensor.detach()) * v) \
                    + 0.5 * self.mse_loss(q_tensor_2 * v, (r + (1 - d) * self.gamma * vp_tensor.detach()) * v)

            loss_critic = (loss_q + loss_v)

            self.optimizer_q.zero_grad()
            loss_critic.backward()
            self.optimizer_q.step()

            # ------------- policy loss ---------------
            mu_prob = dis.normal.Normal(mua_tensor, siga_tensor)

            sampled_u = mu_prob.rsample()
            sampled_a = torch.tanh(sampled_u)

            log_pi_exp = torch.sum(mu_prob.log_prob(sampled_u).clamp(-20, 10), dim=-1,
                                   keepdim=True) - torch.sum(
                torch.log(1 - sampled_a.pow(2) + EPS), dim=-1, keepdim=True)

            loss_a = torch.mean(beta_h * log_pi_exp * v
                                - torch.min(self.f_s2q1(phi_critic.detach(), sampled_a),
                                            self.f_s2q2(phi_critic.detach(), sampled_a)) * v) \
                + REG / 2 * torch.mean(siga_tensor.pow(2) * v.repeat_interleave(self.action_size, dim=-1)
                                       + mua_tensor.pow(2) * v.repeat_interleave(self.action_size, dim=-1))

            if (not self.use_prior_only) and (self.kld_target >= 0) and torch.sum(v).item() > 0:
                kld = (loss_z * self.action_size).data

            elif (not self.use_prior_only) and (self.pq_div_threshold >= 1) and (self.alpha == 0) and torch.sum(v).item() > 0:

                with torch.no_grad():
                    mua_tensor_prior, logsiga_tensor_prior = self.f_s2pi0(hp_tensor)
                    siga_tensor_prior = torch.exp(logsiga_tensor_prior)

                    # ------------- policy loss ---------------
                    mu_prob_prior = dis.normal.Normal(mua_tensor_prior, siga_tensor_prior)

                    sampled_u_prior = mu_prob_prior.rsample()
                    sampled_a_prior = torch.tanh(sampled_u_prior)

                    log_pi_exp_prior = torch.sum(mu_prob_prior.log_prob(sampled_u_prior).clamp(-20, 10), dim=-1,
                                                 keepdim=True) - torch.sum(
                        torch.log(1 - sampled_a_prior.pow(2) + EPS), dim=-1, keepdim=True)

                    loss_a_prior = torch.mean(beta_h * log_pi_exp_prior * v
                                              - torch.min(self.f_s2q1(phi_critic.detach(), sampled_a_prior),
                                                          self.f_s2q2(phi_critic.detach(), sampled_a_prior)) * v) \
                                   + REG / 2 * torch.mean(
                        siga_tensor_prior.pow(2) * v.repeat_interleave(self.action_size, dim=-1)
                        + mua_tensor_prior.pow(2) * v.repeat_interleave(self.action_size, dim=-1))

            self.optimizer_a.zero_grad()
            if not self.use_prior_only:
                (torch.exp(self.log_beta.detach()) * loss_z + loss_a).backward()
                # VLOG for policy
            else:
                loss_a.backward()
            self.optimizer_a.step()

            log_pi_exp_effective = (log_pi_exp * v).detach_().sum() / v.sum()
            # update entropy coefficient if required
            if isinstance(self.beta_h, str):
                self.optimizer_e.zero_grad()

                # if np.random.rand() < 0.01:
                #     print(-log_pi_exp.mean())

                loss_e = - torch.mean(self.log_beta_h * (log_pi_exp_effective + self.target_entropy).detach())
                loss_e.backward()
                self.optimizer_e.step()

                # self.log_beta_h_tar = 1.0 * self.log_beta_h

            # update target Q network
            state_dict_tar = self.target_net.state_dict()
            state_dict = self.state_dict()
            for key in list(self.target_net.state_dict().keys()):
                state_dict_tar[key] = (1 - 1 / self.tau) * state_dict_tar[key] + 1 / self.tau * state_dict[key]
                # state_dict_tar[key] = 0 * state_dict_tar[key] + 1 * state_dict[key]

            self.target_net.load_state_dict(state_dict_tar)

            if (not self.use_prior_only) and (self.kld_target >= 0) and torch.sum(v).item() > 0:
                loss_beta = - torch.mean(self.log_beta * self.alpha * (
                        torch.log10(torch.clamp(kld, 1e-9, np.inf)) - np.log10(self.kld_target)).detach())

                if self.verbose and np.random.rand() < 0.005 * self.verbose:
                    print("log beta = {}, kld = {}, kld_target= {}".format(
                        self.log_beta.item(), kld.detach().item(), self.kld_target))

                self.optimizer_b.zero_grad()
                loss_beta.backward()
                self.optimizer_b.step()

            elif (not self.use_prior_only) and (self.pq_div_threshold >= 1) and (self.alpha == 0) and torch.sum(v).item() > 0:

                self.recent_loss_set[self.update_times % self.recent_loss_set_size] = loss_a.detach().cpu().item()
                self.recent_loss_prior_set[self.update_times % self.recent_loss_set_size] = loss_a_prior.detach().cpu().item()

                fluc_loss = np.std(self.recent_loss_set)
                fluc_loss_prior = np.std(self.recent_loss_prior_set)

                if self.update_times >= self.recent_loss_set_size:
                    if fluc_loss_prior > self.pq_div_threshold * fluc_loss:
                        logbeta_grad = 1
                    elif fluc_loss_prior < fluc_loss / self.pq_div_threshold:
                        logbeta_grad = -1
                    else:
                        logbeta_grad = 0

                    loss_beta = - torch.mean(self.log_beta * logbeta_grad)

                    if self.verbose and np.random.rand() < 0.005 * self.verbose:
                        print("log beta = {}, fluc_loss_q = {}, fluc_loss_q_prior= {}".format(
                            self.log_beta.detach().item(), fluc_loss, fluc_loss_prior))

                    self.optimizer_b.zero_grad()
                    loss_beta.backward()
                    self.optimizer_b.step()

        # --------------- end sac -----------------
        if self.verbose and self.update_times < 10:
            print("training time:", time.time() - start_time)

        self.update_times += 1

        return loss_z.cpu().item(), loss_critic.cpu().item(), loss_a.cpu().item()
