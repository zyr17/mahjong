from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from buffer import MahjongBufferFrost2
import numpy as np
from datetime import datetime
import scipy.special as scisp
import MahjongPy as mp

# ResNet from https://github.com/tensorflow/models/blob/master/official/r1/resnet/resnet_model.py
_BATCH_NORM_DECAY = 0.8
_BATCH_NORM_EPSILON = 1e-2
DEFAULT_VERSION = 2
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES


def batch_norm(inputs, training, data_format):
    """Performs a batch normalization using a standard set of parameters."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops

    return tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=training, fused=True)
    # return inputs



def fixed_padding(inputs, kernel_size, data_format):
    """Pads the input along the spatial dimensions independently of input size.
    Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').
    Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(tensor=inputs,
                             paddings=[[0, 0], [0, 0], [pad_beg, pad_end],
                                       [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(tensor=inputs,
                             paddings=[[0, 0], [pad_beg, pad_end],
                                       [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
    """Strided 2-D convolution with explicit padding."""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)


    return tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format)


def _building_block_v1(inputs, filters, training, projection_shortcut, strides,
                       data_format):
    """A single block for ResNet v1, without a bottleneck.
    Convolution then batch normalization then relu6 as described by:
      Deep Residual Learning for Image Recognition
      https://arxiv.org/pdf/1512.03385.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      filters: The number of filters for the convolutions.
      training: A Boolean for whether the model is in training or inference
        mode. Needed for batch normalization.
      projection_shortcut: The function to use for projection shortcuts
        (typically a 1x1 convolution when downsampling the input).
      strides: The block's stride. If greater than 1, this block will ultimately
        downsample the input.
      data_format: The input format ('channels_last' or 'channels_first').
    Returns:
      The output tensor of the block; shape should match inputs.
    """
    shortcut = inputs

    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
        shortcut = batch_norm(inputs=shortcut, training=training,
                              data_format=data_format)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides,
        data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu6(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=1,
        data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs += shortcut
    inputs = tf.nn.relu6(inputs)

    return inputs


def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides,
                training, name, data_format):
    """Creates one layer of blocks for the ResNet model.
    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      filters: The number of filters for the first convolution of the layer.
      bottleneck: Is the block created a bottleneck block.
      block_fn: The block to use within the model, either `building_block` or
        `bottleneck_block`.
      blocks: The number of blocks contained in the layer.
      strides: The stride to use for the first convolution of the layer. If
        greater than 1, this layer will ultimately downsample the input.
      training: Either True or False, whether we are currently training the
        model. Needed for batch norm.
      name: A string name for the tensor output of the block layer.
      data_format: The input format ('channels_last' or 'channels_first').
    Returns:
      The output tensor of the block layer.
    """

    # Bottleneck blocks end with 4x the number of filters as they start with
    filters_out = filters * 4 if bottleneck else filters

    def projection_shortcut(inputs):
        return conv2d_fixed_padding(
            inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
            data_format=data_format)

    # Only the first block per block_layer uses projection_shortcut and strides
    inputs = block_fn(inputs, filters, training, projection_shortcut, strides,
                      data_format)

    for _ in range(1, blocks):
        inputs = block_fn(inputs, filters, training, None, 1, data_format)

    return tf.identity(inputs, name)


class MahjongNetFrost2():
    """
    Mahjong Network Frost 2
    Using CNN + FC layers, purely feed-forward network
    """
    def __init__(self, graph, agent_no, lr=1e-4, log_dir="../log/", num_tile_type=34, num_each_tile=58, num_vf=30, value_base=10000.0, logging=True):
        """Model function for CNN."""

        self.block_fn = _building_block_v1  # ResNet block function

        self.session = tf.Session(graph=graph)
        self.graph = graph
        self.log_dir = log_dir + ('' if log_dir[-1]=='/' else '/')
        self.value_base = value_base
        self.logging = logging

        self.num_tile_type = num_tile_type  # number of tile types
        self.num_each_tile = num_each_tile # number of features for each tile
        self.num_vf = num_vf # number of vector features

        with self.graph.as_default():

            self.matrix_features = tf.placeholder(dtype=tf.float32, shape=[None, self.num_tile_type, self.num_each_tile], name='matrix_features')
            self.vector_features = tf.placeholder(dtype=tf.float32, shape=[None, self.num_vf], name='vector_features')

            matrix_features = tf.reshape(self.matrix_features, [-1, 1, self.num_tile_type, self.num_each_tile],
                                              name='one_channel_matrix_features')
            self.training = tf.placeholder(dtype=tf.bool, shape=None)

            with tf.variable_scope('trained_net'):
                inputs = conv2d_fixed_padding(
                    inputs=matrix_features,
                    filters=64,
                    kernel_size=3,
                    strides=1,
                    data_format='channels_first')

                inputs = batch_norm(inputs, training=self.training, data_format='channels_first')
                inputs = tf.nn.relu6(inputs)

                inputs = conv2d_fixed_padding(
                    inputs=inputs,
                    filters=64,
                    kernel_size=3,
                    strides=1,
                    data_format='channels_first')
                inputs = batch_norm(inputs, training=self.training, data_format='channels_first')
                inputs = tf.nn.relu6(inputs)

                inputs = conv2d_fixed_padding(
                    inputs=inputs,
                    filters=128,
                    kernel_size=3,
                    strides=3,
                    data_format='channels_first')
                inputs = batch_norm(inputs, training=self.training, data_format='channels_first')
                inputs = tf.nn.relu6(inputs)

                for n in range(1):
                    inputs = conv2d_fixed_padding(
                        inputs=inputs,
                        filters=128,
                        kernel_size=3,
                        strides=1,
                        data_format='channels_first')
                    inputs = batch_norm(inputs, training=self.training, data_format='channels_first')
                    inputs = tf.nn.relu6(inputs)

                inputs = conv2d_fixed_padding(
                    inputs=inputs,
                    filters=256,
                    kernel_size=3,
                    strides=3,
                    data_format='channels_first')
                inputs = batch_norm(inputs, training=self.training, data_format='channels_first')
                inputs = tf.nn.relu6(inputs)

                for n in range(1):
                    inputs = conv2d_fixed_padding(
                        inputs=inputs,
                        filters=256,
                        kernel_size=3,
                        strides=1,
                        data_format='channels_first')
                    inputs = batch_norm(inputs, training=self.training, data_format='channels_first')
                    inputs = tf.nn.relu6(inputs)

                axes = [2, 3]
                inputs = tf.reduce_mean(input_tensor=inputs, axis=axes, keep_dims=True)
                inputs = tf.identity(inputs, 'final_reduce_mean')
                inputs = tf.squeeze(inputs, axes)

                # Dense Layers
                flat = tf.concat([inputs, self.vector_features], axis=1)
                fc1 = tf.layers.dense(inputs=flat, units=512, activation=tf.nn.relu6)
                fc2 = tf.layers.dense(inputs=fc1, units=256, activation=tf.nn.relu6)

                # dropout = tf.layers.dropout(
                #     inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

                self.value_output = tf.layers.dense(inputs=fc2, units=1, activation=None)

            with tf.variable_scope('target_net'):
                inputs = conv2d_fixed_padding(
                    inputs=matrix_features,
                    filters=64,
                    kernel_size=3,
                    strides=1,
                    data_format='channels_first')

                inputs = batch_norm(inputs, training=self.training, data_format='channels_first')
                inputs = tf.nn.relu6(inputs)

                inputs = conv2d_fixed_padding(
                    inputs=inputs,
                    filters=64,
                    kernel_size=3,
                    strides=1,
                    data_format='channels_first')
                inputs = batch_norm(inputs, training=self.training, data_format='channels_first')
                inputs = tf.nn.relu6(inputs)

                inputs = conv2d_fixed_padding(
                    inputs=inputs,
                    filters=128,
                    kernel_size=3,
                    strides=3,
                    data_format='channels_first')
                inputs = batch_norm(inputs, training=self.training, data_format='channels_first')
                inputs = tf.nn.relu6(inputs)

                for n in range(1):
                    inputs = conv2d_fixed_padding(
                        inputs=inputs,
                        filters=128,
                        kernel_size=3,
                        strides=1,
                        data_format='channels_first')
                    inputs = batch_norm(inputs, training=self.training, data_format='channels_first')
                    inputs = tf.nn.relu6(inputs)

                inputs = conv2d_fixed_padding(
                    inputs=inputs,
                    filters=256,
                    kernel_size=3,
                    strides=3,
                    data_format='channels_first')
                inputs = batch_norm(inputs, training=self.training, data_format='channels_first')
                inputs = tf.nn.relu6(inputs)

                for n in range(1):
                    inputs = conv2d_fixed_padding(
                        inputs=inputs,
                        filters=256,
                        kernel_size=3,
                        strides=1,
                        data_format='channels_first')
                    inputs = batch_norm(inputs, training=self.training, data_format='channels_first')
                    inputs = tf.nn.relu6(inputs)

                axes = [2, 3]
                inputs = tf.reduce_mean(input_tensor=inputs, axis=axes, keep_dims=True)
                inputs = tf.identity(inputs, 'final_reduce_mean')
                inputs = tf.squeeze(inputs, axes)

                # Dense Layers
                flat = tf.concat([inputs, self.vector_features], axis=1)
                fc1 = tf.layers.dense(inputs=flat, units=512, activation=tf.nn.relu6)
                fc2 = tf.layers.dense(inputs=fc1, units=256, activation=tf.nn.relu6)

                self.value_output_tarnet = tf.layers.dense(inputs=fc2, units=1, activation=None)

            self.value_target = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='value_targets')

            value_target = self.value_target / self.value_base

            self.loss = tf.losses.mean_squared_error(value_target, self.value_output)
            self.optimizer = tf.train.AdamOptimizer(lr)
            self.train_step = self.optimizer.minimize(self.loss)

            self.saver = tf.train.Saver()

            if self.logging:
                tf.summary.scalar('loss', self.loss)
                tf.summary.histogram('value_pred', self.value_output * self.value_base)

                now = datetime.now()
                datetime_str = now.strftime("%Y%m%d-%H%M%S")

                self.merged = tf.summary.merge_all()
                self.train_writer = tf.summary.FileWriter(log_dir + 'naiveAIlog-Agent{}-'.format(agent_no) + datetime_str, self.session.graph)

            t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
            e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='trained_net')
            self.cg = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
            self.sut = [tf.assign(t, 0.005 * e + 0.995 * t) for t, e in zip(t_params, e_params)]

            self.session.run(tf.global_variables_initializer())
            self.sync()

    def sync(self):
        ## Synchronize target network and trained networ
        self.session.run(self.cg)

    def soft_update_tarnet(self):
        self.session.run(self.sut)

    def save(self, model_dir):
        save_path = self.saver.save(self.session, self.log_dir + model_dir + ('' if model_dir[-1]=='/' else '/') + "FrOst2.ckpt")
        print("Model saved in path: %s" % save_path)

    def restore(self, model_path):
        self.saver.restore(self.session, model_path)
        print("Model restored from" + model_path)

    def output(self, input):
        with self.graph.as_default():
            value_pred = self.session.run(self.value_output,
                                          feed_dict={self.matrix_features: input[0],
                                                     self.vector_features: input[1],
                                                     self.training: False})

        return value_pred * self.value_base

    def output_tarnet(self, input):
        with self.graph.as_default():
            value_pred = self.session.run(self.value_output_tarnet,
                                          feed_dict={self.matrix_features: input[0],
                                                     self.vector_features: input[1],
                                                     self.training: False},)

        return value_pred * self.value_base

    def train(self, input, target, logging, global_step):
        with self.graph.as_default():
            if self.logging and logging:
                loss, _ , summary = self.session.run([self.loss, self.train_step, self.merged],
                                                     feed_dict = {self.matrix_features: input[0],
                                                                  self.vector_features: input[1],
                                                                  self.value_target: target,
                                                                  self.training: True})

                self.train_writer.add_summary(summary, global_step=global_step)
                return loss
            else:
                self.session.run(self.train_step, feed_dict={self.matrix_features: input[0],
                                                             self.vector_features: input[1],
                                                             self.value_target: target,
                                                             self.training: True})
                return None


class AgentFrost2():
    """
    Mahjong AI agent with PER
    """
    def __init__(self, nn: MahjongNetFrost2, memory:MahjongBufferFrost2, gamma=0.9999, greedy=0.03, lambd=0.975, alpha=0.99,
                 num_tile_type=34, num_each_tile=55, num_vf=30):
        self.nn = nn
        self.gamma = gamma  # discount factor
        self.greedy = greedy
        self.memory = memory
        self.lambd = lambd
        self.alpha = alpha # for Ret-GRAPE
        self.global_step = 0
        self.num_tile_type = num_tile_type  # number of tile types
        self.num_each_tile = num_each_tile # number of features for each tile
        self.num_vf = num_vf # number of vector features

        # statistics
        self.stat = {}
        self.stat['greedy'] = self.greedy
        self.stat['num_games'] = 0.
        self.stat['hora_games'] = 0.
        self.stat['ron_games'] = 0.
        self.stat['tsumo_games'] = 0.
        self.stat['fire_games'] = 0.
        self.stat['total_scores_get'] = 0.

        self.stat['hora_rate'] = 0.
        self.stat['tsumo_rate'] = 0.
        self.stat['fire_rate'] = 0.
        self.stat['fulu_rate'] = 0.
        self.stat['riichi_rate'] = 0.
        self.stat['avg_point_get'] = 0.
        self.stat['hora_step'] = 0

        self.stat['hora'] = []
        self.stat['ron'] = []
        self.stat['tsumo'] = []
        self.stat['fire'] = []
        self.stat['score_change'] = []


    def statistics(self, playerNo, result, final_score_change, step, riichi, menchin):

        fulu = 1 - menchin

        if result.result_type == mp.ResultType.RonAgari:
            ron_playerNo = np.argmax(final_score_change)
            if playerNo == ron_playerNo:
                self.stat['hora_games'] += 1
                self.stat['ron_games'] += 1
                self.stat['total_scores_get'] += np.max(final_score_change)
                self.stat['ron'].append(1)
                self.stat['tsumo'].append(0)
                self.stat['hora'].append(1)
                self.stat['fire'].append(0)
            elif riichi * 1000 + final_score_change[playerNo] < 0:
                self.stat['ron'].append(0)
                self.stat['tsumo'].append(0)
                self.stat['hora'].append(0)
                self.stat['fire'].append(1)
                self.stat['fire_games'] += 1
            else:
                self.stat['ron'].append(0)
                self.stat['tsumo'].append(0)
                self.stat['hora'].append(0)
                self.stat['fire'].append(0)

        elif result.result_type == mp.ResultType.TsumoAgari:
            tsumo_playerNo = np.argmax(final_score_change)
            if playerNo == tsumo_playerNo:
                self.stat['hora_games'] += 1
                self.stat['tsumo_games'] += 1
                self.stat['total_scores_get'] += np.max(final_score_change)
                self.stat['ron'].append(0)
                self.stat['tsumo'].append(1)
                self.stat['hora'].append(1)
                self.stat['fire'].append(0)
            else:
                self.stat['ron'].append(0)
                self.stat['tsumo'].append(0)
                self.stat['hora'].append(0)
                self.stat['fire'].append(0)

        else: # RyuuKyoku
            self.stat['ron'].append(0)
            self.stat['tsumo'].append(0)
            self.stat['hora'].append(0)
            self.stat['fire'].append(0)

        self.stat['score_change'].append(final_score_change[playerNo])
        self.stat['num_games'] += 1

        self.stat['hora_rate'] = self.stat['hora_games'] / self.stat['num_games']
        self.stat['tsumo_rate'] = self.stat['tsumo_games'] / self.stat['num_games']
        self.stat['fire_rate'] = self.stat['fire_games'] / self.stat['num_games']

        self.stat['fulu_rate'] = (self.stat['fulu_rate'] * (self.stat['hora_games'] - 1) + riichi) / self.stat['hora_games'] if self.stat['hora_games'] > 0 else 0
        self.stat['riichi_rate'] = (self.stat['riichi_rate'] * (self.stat['hora_games'] - 1) + fulu) / self.stat['hora_games'] if self.stat['hora_games'] > 0 else 0

        self.stat['hora_step'] = (self.stat['hora_step'] * (self.stat['hora_games'] - 1) + step) / self.stat['hora_games'] if self.stat['hora_games'] > 0 else 0
        self.stat['avg_point_get'] = self.stat['total_scores_get'] / self.stat['hora_games'] if self.stat['hora_games'] > 0 else 0


    def select(self, aval_next_states):
        """
        select an action according to the value estimation of the next state after performing an action
        :param:
            aval_next_states: (matrix_features [N by self.num_tile_type by self.num_each_tile], vector features [N by self.num_vf]), where N is number of possible
                actions (corresponds to aval_next_states)
        :return:
            action: an int number, indicating the index of being selected action,from [0, 1, ..., N]
            policy: an N-dim vector, indicating the probabilities of selecting actions [0, 1, ..., N]
        """
        if aval_next_states is None:
            return None

        aval_next_matrix_features = np.reshape(aval_next_states[0], [-1, self.num_tile_type, self.num_each_tile])
        aval_next_vector_features = np.reshape(aval_next_states[1], [-1, self.num_vf])

        next_value_pred = np.reshape(self.nn.output((aval_next_matrix_features, aval_next_vector_features)), [-1])

        # softmax policy
        # policy = scisp.softmax(self.greedy * next_value_pred)
        #
        # policy /= policy.sum()
        # action = np.random.choice(np.size(policy), p=policy)

        policy = np.zeros_like(next_value_pred, dtype=np.float32)

        ind = np.argmax(next_value_pred)
        policy[ind] = 1
        policy = policy + self.greedy
        policy = policy / np.sum(policy)

        action = np.random.choice(np.size(policy), p=policy)

        return action, policy

    def remember_episode(self, num_aval_actions, next_matrix_features, next_vector_features, rewards, dones, actions, behavior_policies, weight):
        # try:

        if len(dones) == 0:
            print("Episode Length 0! Not recorded!")
        else:
            self.memory.append_episode(num_aval_actions,
                                       next_matrix_features,
                                       next_vector_features,
                                       np.reshape(rewards, [-1,]),
                                       np.reshape(dones, [-1,]),
                                       np.reshape(actions, [-1]),
                                       np.reshape(behavior_policies, [-1, 40]),
                                       weight=weight)
        # except:
        #     print("Episode Length 0! Not recorded!")
    def learn(self, symmetric_hand=None, sequence_length=32, batch_size=8, episode_start=1, care_lose=True, logging=True):

        if self.memory.filled_size >= episode_start:

            all_Sp = []
            all_sp = []
            all_target_value = []

            for b in range(batch_size):
                n_t, Sp, sp, r_t, done_t, a_t, mu_t, length, e_index, e_weight = self.memory.sample_episode()

                start_t = np.random.randint(1 - sequence_length, length - 1)
                sampled_ts = np.arange(max(0, start_t), min(length, start_t + sequence_length))

                l = len(sampled_ts)
                n_t = n_t[sampled_ts]
                Sp = Sp[sampled_ts]
                sp = sp[sampled_ts]
                r_t = r_t[sampled_ts]
                done_t = done_t[sampled_ts]
                a_t = a_t[sampled_ts]
                mu_t = mu_t[sampled_ts]

                # this_Sp = np.zeros([l, self.num_tile_type, self.num_each_tile], dtype=np.float32)
                # this_sp = np.zeros([l, self.num_vf], dtype=np.float32)

                this_Sp = Sp[np.arange(l), a_t].astype(np.float32)
                this_sp = sp[np.arange(l), a_t].astype(np.float32)

                if not care_lose:
                    r_t = np.maximum(r_t, 0)

                mu_size = mu_t.shape[1]

                # _, policy_all = self.select((Sp.reshape([-1, self.num_tile_type, self.num_each_tile]), sp.reshape([-1, self.num_vf])))
                # pi = policy_all.reshape([l, -1])

                q_tar = self.nn.output_tarnet(
                    (Sp.reshape([-1, self.num_tile_type, self.num_each_tile]), sp.reshape([-1, self.num_vf]))).reshape(
                    [l, mu_size])
                q = self.nn.output(
                    (Sp.reshape([-1, self.num_tile_type, self.num_each_tile]), sp.reshape([-1, self.num_vf]))).reshape(
                    [l, mu_size])

                for t in range(l):
                    q[t, n_t[t]:] = - np.inf  # for computing policy
                    q_tar[t, n_t[t]:] = 0

                # pi = scisp.softmax(q, axis=1)  # to get the true pi

                pi = np.zeros_like(q, dtype=np.float32)

                for tau in range(pi.shape[0]):
                    ind = np.argmax(q[tau, :n_t[tau]])
                    pi[tau, ind] = 1
                    pi[tau, :n_t[tau]] += self.greedy
                    pi[tau] = pi[tau] / np.sum(pi[tau])

                pi_t, pi_tp1 = pi, np.concatenate((pi[1:, :], np.zeros([1, mu_size])), axis=0)
                q_t, q_tp1 = q_tar, np.concatenate((q_tar[1:, :], np.zeros([1, mu_size])), axis=0)
                q_t_a = q_t[np.arange(l), a_t]
                v_t, v_tp1 = np.sum(pi_t * q_t, axis=1), np.sum(pi_tp1 * q_tp1, axis=1)
                q_t_a_est = r_t + (1. - done_t) * self.gamma * v_tp1
                td_error = q_t_a_est - q_t_a + self.alpha * (q_t_a - v_t)
                rho_t_a = pi_t[np.arange(l), a_t] / mu_t[np.arange(l), a_t]   # importance sampling ratios
                c_t_a = self.lambd * np.minimum(rho_t_a, 1)

                # print('td_eror')
                # print(td_error[-5:])

                y_prime = 0  # y'_t
                g_q = np.zeros([l])
                for u in reversed(range(l)):  # l-1, l-2, l-3, ..., 0
                    # If s_tp1[u] is from an episode different from s_t[u], y_prime needs to be reset.

                    y_prime = 0 if done_t[u] else y_prime  # y'_u
                    g_q[u] = q_t_a_est[u] + y_prime

                    # y'_{u-1} used in the next step
                    y_prime = self.lambd * self.gamma * np.minimum(rho_t_a[u], 1) * td_error[u] + self.gamma * c_t_a[u] * y_prime

                target_q = g_q + self.alpha * (q_t_a - v_t)
                target_q = target_q.reshape([l, 1])

                if not symmetric_hand == None:
                    all_Sp.append(symmetric_hand(this_Sp))
                    all_sp.append(this_sp)
                    all_target_value.append(target_q)
                else:
                    all_Sp.append(this_Sp)
                    all_sp.append(this_sp)
                    all_target_value.append(target_q)

            # this_Sp = np.zeros([r.shape[0], self.num_tile_type, self.num_each_tile], dtype=np.float32)
            # this_sp = np.zeros([r.shape[0], self.num_vf], dtype=np.float32)
            # target_q = np.zeros([r.shape[0], 1], dtype=np.float32)
            #
            # episode_length = r.shape[0]
            #
            # td_prime = 0
            #
            # q_all = self.nn.output((Sp, sp))
            #
            # _, policy_all = self.select((Sp.reshape([-1, self.num_tile_type, self.num_each_tile]), sp.reshape([-1, self.num_vf])))
            # policy_all = policy_all.reshape([episode_length, -1])
            #
            # for t in reversed(range(episode_length)):  #Q(lambda)
            #
            #     this_Sp[t] = Sp[t, a[t]]
            #     this_sp[t] = sp[t, a[t]]
            #
            #     q_all_t = q_all[t, 0:n[t]]
            #     q_t_a = q_all_t[a[t], :]
            #
            #     mu_t_a = mu[t, a[t]]
            #     pi_t_a = policy_all[t, a[t]]
            #
            #     if d[t]:
            #         q_t_a_est = r[t]
            #     else:
            #         q_all_tp1 = q_all[t+1, 0:n[t+1]]
            #         policy_tp1 = policy_all[t+1, 0:n[t+1]]
            #         v_tp1 = np.sum(policy_tp1.reshape([n[t+1]]) * q_all_tp1.reshape([n[t+1]]))
            #         q_t_a_est = r[t] + self.gamma * v_tp1
            #
            #     rho_t_a = pi_t_a / mu_t_a
            #     c_t_a = self.lambd * np.minimum(rho_t_a, 1)
            #
            #     td_error_t = q_t_a_est - q_t_a
            #
            #     if d[t]:
            #         td_prime = 0
            #     else:
            #         td_prime = td_prime
            #
            #     target_q[t] = q_t_a_est + td_prime
            #
            #     td_prime = self.gamma * c_t_a * (td_error_t + td_prime)

            # print('target_q')
            # print(target_q[-5:,0])

            self.global_step += 1
            all_Sp = np.vstack(all_Sp)
            all_sp = np.vstack(all_sp)
            all_target_value = np.vstack(all_target_value)

            # also train symmetric hand
            # if not symmetric_hand == None:
            #     all_Sp = np.concatenate([symmetric_hand(this_Sp) for _ in range(5)], axis=0).astype(np.float32)
            #     all_sp = np.concatenate([this_sp for _ in range(5)], axis=0).astype(np.float32)
            #     all_target_value = np.concatenate([target_q for _ in range(5)], axis=0).astype(np.float32)
            # else:
            #     all_Sp = this_Sp
            #     all_sp = this_sp
            #     all_target_value = target_q

            self.nn.train((all_Sp, all_sp), all_target_value, logging=logging, global_step=self.global_step)
            self.nn.soft_update_tarnet()

        else:
            pass

