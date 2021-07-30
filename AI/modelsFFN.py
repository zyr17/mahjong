import warnings

import gym
import torch
import torch.nn as nn
import numpy as np
import time
from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributions as dis
from gym.spaces import Box, Discrete

from bn_net import *

torch.set_default_dtype(torch.float32)

INFINITY = 1e10


def softplus_exp(x):
    y = torch.log(1 + torch.exp(x))
    return y


class VLOGFFN(nn.Module):

    def __init__(self, observation_space, full_observation_space, action_space, is_main_network=True, **kwargs):

        super(VLOGFFN, self).__init__()

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

        if len(self.input_forward_size) in [2, 3]:
            # TODO: when oracle observation has different shape
            self.image_input = True

            if len(self.input_forward_size) == 2:
                resolution = "{}".format(self.input_forward_size[1])
            else:
                resolution = "{}x{}".format(self.input_forward_size[1], self.input_forward_size[2])

            self.encoder, phi_size = make_cnn(resolution, self.input_forward_size[0], batch_norm_tau=self.batch_norm_tau)

            self.encoder_oracle, phi_size_oracle = make_cnn(resolution, self.input_oracle_size[0], batch_norm_tau=self.batch_norm_tau)

            self.encoder_critic, phi_size_oracle = make_cnn(resolution, self.input_oracle_size[0], batch_norm_tau=self.batch_norm_tau)
            # only used for actor critic

            self.phi_size = int(phi_size)
            self.phi_size_oracle = int(phi_size_oracle)

        elif len(self.input_forward_size) == 1:
            if self.batch_norm_tau:
                raise NotImplementedError
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
            self.phi_size = self.hidden_layer_width
            self.phi_size_oracle = self.hidden_layer_width
        else:
            raise NotImplementedError("Observation space must be either 2-D, 3-D (image-like), or 1-D (vector)!")

        self.latent_module.append(self.encoder_oracle)
        self.latent_module.append(self.encoder)

        forward_fnns = nn.ModuleList()
        last_layer_size = self.phi_size
        for _ in range(self.half_hidden_layer_depth - 1):
            forward_fnns.append(nn.Linear(last_layer_size, self.hidden_layer_width))
            if self.batch_norm_tau:
                forward_fnns.append(BatchNorm1d_LastDim(num_features=self.hidden_layer_width, momentum=1 / self.batch_norm_tau))
            forward_fnns.append(self.forward_act_fn())
            last_layer_size = self.hidden_layer_width
        self.forward_fnn = nn.Sequential(*forward_fnns)
        self.latent_module.append(self.forward_fnn)

        pre_zp_size = self.hidden_layer_width

        if not self.z_size == 0:
            if self.z_deterministic_size:
                if self.batch_norm_tau:
                    self.f_h2zp_det = nn.Sequential(nn.Linear(pre_zp_size, self.z_deterministic_size),
                                                    BatchNorm1d_LastDim(num_features=self.z_deterministic_size,
                                                                   momentum=1 / self.batch_norm_tau),
                                                    )
                else:
                    self.f_h2zp_det = nn.Sequential(nn.Linear(pre_zp_size, self.z_deterministic_size)
                                                    )
                self.latent_module.append(self.f_h2zp_det)
            if self.z_stochastic_size:

                if self.batch_norm_tau:
                    self.f_h2muzp = nn.Sequential(nn.Linear(pre_zp_size, self.z_stochastic_size),
                                                  BatchNorm1d_LastDim(num_features=self.z_stochastic_size,
                                                                 momentum=1 / self.batch_norm_tau)
                                                  )

                    self.f_h2logsigzp = nn.Sequential(nn.Linear(pre_zp_size, self.z_stochastic_size),
                                                      BatchNorm1d_LastDim(num_features=self.z_stochastic_size,
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
                oracle_fnns.append(BatchNorm1d_LastDim(num_features=self.hidden_layer_width, momentum=1 / self.batch_norm_tau))
            oracle_fnns.append(self.forward_act_fn())
            last_layer_size = self.hidden_layer_width

        self.oracle_fnn = nn.Sequential(*oracle_fnns)
        self.latent_module.append(self.oracle_fnn)

        if not self.z_size == 0:
            pre_zq_size = self.hidden_layer_width

            if self.z_deterministic_size:
                if self.batch_norm_tau:
                    self.f_hb2zq_det = nn.Sequential(nn.Linear(pre_zq_size, self.z_deterministic_size),
                                                     BatchNorm1d_LastDim(num_features=self.z_deterministic_size, momentum=1 / self.batch_norm_tau)
                                                     )
                else:
                    self.f_hb2zq_det = nn.Sequential(nn.Linear(pre_zq_size, self.z_deterministic_size)
                                                     )
                self.latent_module.append(self.f_hb2zq_det)

            if self.z_stochastic_size:
                if self.batch_norm_tau:
                    self.f_hb2muzq = nn.Sequential(nn.Linear(pre_zq_size, self.z_stochastic_size),
                                                   BatchNorm1d_LastDim(num_features=self.z_stochastic_size, momentum=1 / self.batch_norm_tau)
                                                   )

                    self.f_hb2logsigzq = nn.Sequential(nn.Linear(pre_zq_size, self.z_stochastic_size),
                                                       BatchNorm1d_LastDim(num_features=self.z_stochastic_size, momentum=1 / self.batch_norm_tau),
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

            self.optimizer_q = torch.optim.Adam([*self.f_s2q1.parameters(), *self.f_s2q2.parameters(),
                                                 *self.f_s2v.parameters(), *self.encoder_critic.parameters()],
                                                lr=self.lr)
            self.optimizer_a = torch.optim.Adam([*self.latent_module.parameters(), *self.f_s2pi0.parameters()],
                                                lr=self.lr)

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

            # Q network 1
            self.f_s2q1 = DiscreteActionQNetwork(pre_rl_size, self.action_size,
                                                 hidden_layers=[self.hidden_layer_width] * self.half_hidden_layer_depth,
                                                 dueling=self.dueling, output_distribution=self.value_distribution,
                                                 act_fn=self.forward_act_fn)

            self.optimizer_q = torch.optim.Adam(self.parameters(), lr=self.lr)

        else:
            raise NotImplementedError("algorithm can only be 'sac' or 'ddqn'")

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

            target_net = VLOGFFN(observation_space, full_observation_space, action_space, is_main_network=False, **kwargs)

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

    def learn(self, X, X_oracle, A, R, D, V, L, seq_len=8, batch_size=32, action_masks=None, gradient_steps=None):
        self.train()
        self.target_net.eval()

        if gradient_steps is None:
            gradient_steps = batch_size * seq_len

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

        results = None

        normal_update_ratio = 1 / (1 + self.tabular_like_reg)

        for _ in range(gradient_steps):

            if np.random.rand() < normal_update_ratio:
                results = self.train_rl(x_seq_batch, x_oracle_seq_batch, action_seq_batch, reward_seq_batch, done_seq_batch,
                                        valid_seq_batch, mask_seq_batch, seq_len, batch_size, is_tblr_step=False)
            else:
                # Resample other data
                e_samples = np.random.choice(len(L), batch_size,
                                             p=weights / weights.sum())  # sample with equal p for each step

                # print(e_samples)
                x_seq_tblr = [X[e_samples[i]] for i in range(batch_size)]
                x_oracle_seq_tblr = [X_oracle[e_samples[i]] for i in range(batch_size)]
                action_seq_tblr = [A[e_samples[i]] for i in range(batch_size)]
                reward_seq_tblr = [R[e_samples[i]] for i in range(batch_size)]
                done_seq_tblr = [D[e_samples[i]] for i in range(batch_size)]

                if V is not None:
                    valid_seq_tblr = [V[e_samples[i]] for i in range(batch_size)]
                else:
                    valid_seq_tblr = [np.ones_like(D[e_samples[i]]) for i in range(batch_size)]

                if action_masks is not None:
                    mask_seq_tblr = [action_masks[e_samples[i]] for i in range(batch_size)]
                else:
                    mask_seq_tblr = None

                self.train_rl(x_seq_tblr, x_oracle_seq_tblr, action_seq_tblr, reward_seq_tblr,
                              done_seq_tblr, valid_seq_tblr, mask_seq_tblr, seq_len, batch_size, is_tblr_step=True)

        self.eval()
        return results

    def train_rl(self, x_seq_batch, x_oracle_seq_batch, action_seq_batch, reward_seq_batch, done_seq_batch,
                 valid_seq_batch, mask_seq_batch, seq_len, batch_size, is_tblr_step=False):

        # ------------ pre-process numpy data ------------

        start_time = time.time()

        x, xp, x_oracle, xp_oracle, a, r, d, v, m = \
            self.preprocess_data_fnn(x_seq_batch, x_oracle_seq_batch, action_seq_batch, reward_seq_batch,
                                     done_seq_batch, valid_seq_batch, seq_len, mask_seq_batch=mask_seq_batch)  # Here is seq_len, no need +1
        # print(x.shape)

        self.train()
        self.target_net.eval()

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

                loss_z = torch.mean(
                    torch.sum(logsigzp_tensor - logsigzq_tensor + ((muzp_tensor - muzq_tensor).pow(2) + torch.exp(
                        logsigzq_tensor * 2)) / (2.0 * torch.exp(logsigzp_tensor * 2)) - 0.5, dim=-1) * v)

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

            # ---------- Compute Q Target ---------------
            self.eval()

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

            self.train()
            self.target_net.eval()

            if is_tblr_step:
                loss_critic = self.tabular_like_reg * torch.mean(torch.sum(
                    - self.f_s2q1.get_log_prob(h_tp1_tensor, qp_tensor_tar.detach()) * a_one_hot, dim=-1) * v)

            else:
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

        elif self.algorithm == "sac":

            if self.batch_norm_tau != 0:
                raise NotImplementedError

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

            if is_tblr_step:

                q_tensor_tar_1 = self.target_net.f_s2q1(phi_critic_tar, a).detach()
                q_tensor_tar_2 = self.target_net.f_s2q2(phi_critic_tar, a).detach()

                loss_q = self.tabular_like_reg * (0.5 * self.mse_loss(q_tensor_1 * v, q_tensor_tar_1 * v)
                                                  + 0.5 * self.mse_loss(q_tensor_2 * v, q_tensor_tar_2 * v))
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

        # --------------- end sac -----------------
        if self.verbose and self.update_times < 10:
            print("training time:", time.time() - start_time)

        self.update_times += 1

        return loss_z.cpu().item(), loss_critic.cpu().item(), loss_a.cpu().item()
