#!/usr/bin/env python
# coding: utf-8

from wrapper import *
import MahjongPy as mp
import os
import time
import logging
import warnings

import numpy as np
import scipy.io as sio
import torch
import matplotlib.pyplot as plt


from models import VLOG



UNICODE_TILES = """
    🀇 🀈 🀉 🀊 🀋 🀌 🀍 🀎 🀏 
    🀙 🀚 🀛 🀜 🀝 🀞 🀟 🀠 🀡
    🀐 🀑 🀒 🀓 🀔 🀕 🀖 🀗 🀘
    🀀 🀁 🀂 🀃
    🀆 🀅 🀄
""".split()

EXPLAINS = UNICODE_TILES + ["Chi-Left", "Chi-Middle", "Chi-Right", "Pon", "An-Kan",
           "Min-Kan", "Add-Kan", "Riichi", "Ron", "Tsumo", "Push 99", "Pass"]


def to_unicode_tails(tiles, sort=True):
    unicode_tail_string = ""
    if sort:
        tiles = np.sort(np.array(tiles))

    for tile in tiles:
        unicode_tail_string = unicode_tail_string + UNICODE_TILES[int(tile / 4)]

    return unicode_tail_string


# In[3]:


env_test = EnvMahjong3()
max_steps = 1000


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


agent_test0 = torch.load("/mnt/c/code/snake/amlt/vlog_offlinemj_bn_20210728/search_envid_1_beta_0_alpha_1_kldt_-1_bnt_0_cqla_100/data/mahjong_VLOG_DDQN_0.model", map_location=torch.device(device))
agent_test1 = torch.load("/mnt/c/code/snake/amlt/vlog_offlinemj_bn_20210728/search_envid_1_beta_0_alpha_1_kldt_-1_bnt_0_cqla_100/data/mahjong_VLOG_DDQN_0.model", map_location=torch.device(device))
agent_test2 = torch.load("/mnt/c/code/snake/amlt/vlog_offlinemj_bn_20210728/search_envid_1_beta_0_alpha_1_kldt_-1_bnt_0_cqla_100/data/mahjong_VLOG_DDQN_0.model", map_location=torch.device(device))
# agent_test3 = torch.load("./vlog_offlinemj_20210724/search_envid_1_beta_0_alpha_1_kldt_-1_cqla_100/data/mahjong_VLOG_DDQN_3.model", map_location=torch.device(device))


agents_test = [agent_test0, agent_test1, agent_test2, agent_test0]

for agent in agents_test:
    agent.device = device

full_obs = False


# In[ ]:


EpiTestRet = 0
steps_taken = 0

for game in range(1000):
    print("==============================Game {}=============================".format(game))
    # reset each episode
    sp = env_test.reset(0, 'east')
    done = False

    for tt in range(max_steps):

        # if curr_pid != 0:
        #     a = valid_actions[np.random.randint(len(valid_actions))]

        valid_actions = env_test.get_valid_actions(nhot=False)

        if len(valid_actions) == 1:
            env_test.t.make_selection(0)

        else:
            curr_pid = env_test.get_curr_player_id()

            action_mask = env_test.get_valid_actions(nhot=True)

            if not full_obs:
                a = agents_test[curr_pid].select(env_test.get_obs(curr_pid), action_mask, greedy=True)
            else:
                a = agents_test[curr_pid].select(env_test.get_full_obs(curr_pid), action_mask, greedy=True)

            if curr_pid == 0 and len(valid_actions) > 1 and a > 40 and a < 45:
                print("-------------- Step {}, player {} ----------------".format(tt, curr_pid))
                print(env_test.Phases[env_test.t.get_phase()], "Recent Tile:", to_unicode_tails([env_test.latest_tile]))
                side_tiles_0 = [st[0] for st in env_test.side_tiles[curr_pid]]
                print("手牌: ", to_unicode_tails(env_test.hand_tiles[curr_pid]),
                      "； 副露：", to_unicode_tails(side_tiles_0))
                if a < 34:
                    agent_selection_str = UNICODE_TILES[a]
                else:
                    agent_selection_str = EXPLAINS[a]

                print("Agent选择打: ", agent_selection_str)

            sp, r, done, _ = env_test.step(curr_pid, a)

        steps_taken += 1

        if done or env_test.t.get_phase() == 16:
            payoffs = env_test.get_payoffs()
            # EpiTestRet += payoffs[0]
            
            print("~~~~~~~~~~~~Result: ", payoffs)

            print(env_test.t.game_log.to_string())
            # if np.max(payoffs) > 0:
            #     plt.pcolor(sp)
            #     plt.show()
            #     print(env_test.Phases[env_test.t.get_phase()], "Recent Tile:", to_unicode_tails([env_test.latest_tile]))
            #     side_tiles_0 = [st[0] for st in env_test.side_tiles[np.argmax(payoffs)]]
            #     print("手牌: ", to_unicode_tails(env_test.hand_tiles[np.argmax(payoffs)]),
            #       "； 副露：", to_unicode_tails(side_tiles_0))
            break

        # if not done and env_test.get_curr_player_id() == 0 and len(valid_actions) > 1:
        #     shanten_num = np.zeros(4)
        #     for pid in range(4):
        #         shanten_num[pid] = shanten.calculate_shanten(TilesConverter.to_34_array(
        #             env_test.hand_tiles[pid] + [st[0] for st in env_test.side_tiles[pid]]))
        #     print("向听数:", shanten_num)

