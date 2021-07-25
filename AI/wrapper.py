import platform

import numpy as np
from copy import deepcopy
import gym
import MahjongPy as mp

from gym.spaces import Discrete, Box

player_i_hand_start_ind = [0, 63, 69, 75]  # later 3 in oracle_obs
player_i_side_start_ind = [6, 12, 18, 24]
player_i_river_start_ind = [30, 37, 44, 51]

dora_indicator_ind = 58
dora_ind = 59
game_wind_ind = 60
self_wind_ind = 61
latest_tile_ind = 62

aka_tile_ints = [16, 16 + 36, 16 + 36 + 36]
player_obs_width = 63

CHILEFT = 34
CHIMIDDLE = 35
CHIRIGHT = 36
PON = 37
ANKAN = 38
MINKAN = 39
ADDKAN = 40

RIICHI = 41
RON = 42
TSUMO = 43
PUSH = 44
NOOP = 45

UNICODE_TILES = """
    🀇 🀈 🀉 🀊 🀋 🀌 🀍 🀎 🀏 
    🀙 🀚 🀛 🀜 🀝 🀞 🀟 🀠 🀡
    🀐 🀑 🀒 🀓 🀔 🀕 🀖 🀗 🀘
    🀀 🀁 🀂 🀃
    🀆 🀅 🀄
""".split()


def dora_ind_2_dora_id(ind_id):
    if ind_id == 8:
        return 0
    elif ind_id == 8 + 9:
        return 0 + 9
    elif ind_id == 8 + 9 + 9:
        return 0 + 9 + 9
    elif ind_id == 30:
        return 27
    elif ind_id == 33:
        return 31
    else:
        return ind_id + 1


def dora2indicator(dora_id):
    if dora_id == 0:  # 1m
        indicator_id = 8  # 9m
    elif dora_id == 9:  # 1p
        indicator_id = 17  # 9p
    elif dora_id == 18:  # 1s
        indicator_id = 26  # 9s
    elif dora_id == 27:  # East
        indicator_id = 30  # North
    elif dora_id == 31:  # Hake
        indicator_id = 33  # Chu
    else:
        indicator_id = dora_id - 1
    return indicator_id


def is_consecutive(a: int, b: int, c: int):
    array = np.array([a, b, c])
    return array[1] - array[0] == 1 and array[2] - array[1] == 1


def generate_obs(hand_tiles, river_tiles, side_tiles, dora_tiles, game_wind, self_wind, latest_tile=None):
    all_obs_0p = np.zeros([34, 63 + 18], dtype=np.uint8)

    global player_i_hand_start_ind
    global player_i_side_start_ind
    global player_i_river_start_ind

    global dora_indicator_ind
    global dora_ind
    global game_wind_ind
    global self_wind_ind
    global latest_tile_ind

    global aka_tile_ints

    # ----------------- Side Tiles Process ------------------
    for player_id, player_side_tiles in enumerate(side_tiles):
        side_tile_num = np.zeros(34, dtype=np.uint8)
        for side_tile in player_side_tiles:
            side_tile_id = int(side_tile[0] / 4)
            side_tile_num[side_tile_id] += 1

            if side_tile[0] in aka_tile_ints:
                # Red dora
                all_obs_0p[side_tile_id, player_i_side_start_ind[player_id] + 5] = 1
            if side_tile[1]:
                all_obs_0p[side_tile_id, player_i_side_start_ind[player_id] + 4] = 1

        for t_id in range(34):
            for k in range(4):
                if side_tile_num[t_id] > k:
                    all_obs_0p[t_id, player_i_side_start_ind[player_id] + k] = 1

    # ----------------- River Tiles Procces ------------------
    for player_id, player_river_tiles in enumerate(river_tiles):  # 副露也算在牌河里, also include Riichi info
        river_tile_num = np.zeros(34, dtype=np.uint8)
        for river_tile in player_river_tiles:
            river_tile_id = int(river_tile[0] / 4)

            all_obs_0p[river_tile_id, player_i_hand_start_ind[player_id] + 4] = 1

            river_tile_num[river_tile_id] += 1

            if river_tile[0] in aka_tile_ints:  # Red dora
                all_obs_0p[river_tile_id, player_i_river_start_ind[player_id] + 5] = 1

            # te-kiri (from hand)
            all_obs_0p[river_tile_id, player_i_river_start_ind[player_id] + 4] += river_tile[1]

            # is riichi-announcement tile
            all_obs_0p[river_tile_id, player_i_river_start_ind[player_id] + 6] += river_tile[2]

        for t_id in range(34):
            for k in range(4):
                if river_tile_num[t_id] > k:
                    all_obs_0p[t_id, player_i_river_start_ind[player_id] + k] = 1

    # ----------------- Hand Tiles Process ------------------
    for player_id, player_hand_tiles in enumerate(hand_tiles):
        hand_tile_num = np.zeros(34, dtype=np.uint8)
        for hand_tile in player_hand_tiles:
            hand_tile_id = int(hand_tile / 4)
            hand_tile_num[hand_tile_id] += 1

            if hand_tile in aka_tile_ints:
                # Aka dora
                all_obs_0p[hand_tile_id, player_i_hand_start_ind[player_id] + 5] = 1

            # how many times this tile has been discarded before by this player
            all_obs_0p[hand_tile_id, player_i_hand_start_ind[player_id] + 4] = (np.sum(
                all_obs_0p[hand_tile_id,
                player_i_river_start_ind[player_id]:player_i_river_start_ind[player_id] + 4])) > 0

        for t_id in range(34):
            for k in range(4):
                if hand_tile_num[t_id] > k:
                    all_obs_0p[t_id, player_i_hand_start_ind[player_id] + k] = 1

    # ----------------- Dora Process ------------------
    for dora_tile in dora_tiles:
        dora_hai_id = int(dora_tile / 4)
        all_obs_0p[dora_hai_id, dora_ind] += 1
        all_obs_0p[dora2indicator(dora_hai_id), dora_indicator_ind] += 1

    # ----------------- Public Game State ----------------
    all_obs_0p[:, game_wind_ind] = game_wind  # Case 1 to 4 in dim 0
    all_obs_0p[:, self_wind_ind] = self_wind

    #------------ Latest Tile -------------
    if latest_tile is not None:
        all_obs_0p[int(latest_tile / 4), latest_tile_ind] = 1

    # players_obs = all_obs_0p[:, :63]
    # oracles_obs = all_obs_0p[:, 63:]

    return all_obs_0p


class EnvMahjong3(gym.Env):
    """
    Mahjong Environment for FrOst Ver3
    """

    metadata = {'name': 'Mahjong', 'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}
    spec = {'id': 'TaskT'}

    def __init__(self, printing=True, reward_unit=100):
        self.t = mp.Table()
        self.Phases = (
            "P1_ACTION", "P2_ACTION", "P3_ACTION", "P4_ACTION", "P1_RESPONSE", "P2_RESPONSE", "P3_RESPONSE",
            "P4_RESPONSE", "P1_抢杠RESPONSE", "P2_抢杠RESPONSE", "P3_抢杠RESPONSE", "P4_抢杠RESPONSE",
            "P1_抢暗杠RESPONSE", "P2_抢暗杠RESPONSE", " P3_抢暗杠RESPONSE", " P4_抢暗杠RESPONSE", "GAME_OVER",
            "P1_DRAW, P2_DRAW, P3_DRAW, P4_DRAW")
        self.horas = [False, False, False, False]
        self.played_a_tile = [False, False, False, False]
        self.tile_in_air = None
        self.final_score_changes = []
        self.game_count = 0
        self.printing = printing
        self.reward_unit = reward_unit

        self.vector_feature_size = 30

        self.scores_init = np.zeros([4], dtype=np.float32)
        for i in range(4):
            self.scores_init[i] = self.t.players[i].score

        self.scores_before = np.zeros([4], dtype=np.float32)
        for i in range(4):
            self.scores_before[i] = self.t.players[i].score

        self.observation_space = Box(low=0, high=4, shape=[63, 34])
        self.full_observation_space = Box(low=0, high=4, shape=[81, 34])

        self.action_space = Discrete(46)

    def reset(self, oya, game_wind):
        self.t = mp.Table()

        # oya = np.random.random_integers(0, 3)
        # winds = ['east', 'south', 'west', 'north']
        oya = '{}'.format(oya)

        self.oya_id = int(oya)

        # self.t.game_init()
        self.t.game_init_with_metadata({"oya": oya, "wind": game_wind})

        # ----------------- Statistics ------------------
        self.game_count += 1

        self.horas = [False, False, False, False]
        self.played_a_tile = [False, False, False, False]

        self.scores_init = np.zeros([4], dtype=np.float32)
        for i in range(4):
            self.scores_init[i] = self.t.players[i].score

        self.scores_before = np.zeros([4], dtype=np.float32)
        for i in range(4):
            self.scores_before[i] = self.t.players[i].score

        # ---------------- Table raw state ----------------
        self.curr_all_obs = np.zeros([4, 34, 81])

        self.hand_tiles = [[], [], [], []]
        self.river_tiles = [[], [], [], []]
        self.side_tiles = [[], [], [], []]
        self.dora_tiles = []
        self.game_wind_obs = np.zeros(34)  # index: -4

        self.dora_tiles.append(dora_ind_2_dora_id(int(self.t.DORA[0].tile)))

        if game_wind == "east":
            self.game_wind_obs[27] = 1
        elif game_wind == "south":
            self.game_wind_obs[28] = 1
        elif game_wind == "west":
            self.game_wind_obs[29] = 1
        elif game_wind == "north":
            self.game_wind_obs[30] = 1
        else:
            raise ValueError

        self.latest_tile = None

        for pid in range(4):
            for i in range(len(self.t.players[pid].hand)):
                self.hand_tiles[pid].append(int(self.t.players[pid].hand[i].tile))

        who, what = self.who_do_what()

        return self.get_obs(who)

    def get_valid_actions(self, nhot=True):
        if not nhot:
            return self.curr_valid_actions
        else:
            curr_valid_actions_mask = np.zeros(self.action_space.n, dtype=np.bool)
            curr_valid_actions_mask[self.curr_valid_actions] = 1
            return curr_valid_actions_mask

    def get_curr_player_id(self):
        who, what = self.who_do_what()
        return who

    def get_payoffs(self):
        scores_change = self.final_score_changes()
        payoff = [sc / self.reward_unit for sc in scores_change]
        return payoff

    def is_over(self):
        return self.Phases[self.t.get_phase()] == "GAME_OVER"

    def step(self, player_id, action, raw_action=False):
        who, what = self.who_do_what()

        assert who == player_id

        valid_actions = self.get_valid_actions(nhot=False)
        if action not in valid_actions:
            raise ValueError("action is not valid!! \
                Current valid actions can be obtained by env.get_valid_actions(onehot=False)")

        assert len(np.argwhere(valid_actions == action)) == 1
        action_no = np.argwhere(valid_actions == action)[0]

        if what == "reponse":
            return self.step_response(action_no, player_id)
        elif what == "play":
            return self.step_play(action_no, player_id)

    def get_obs(self, player_id):

        player_wind_obs = np.zeros([34])
        player_wind_obs[27 + (8 - self.oya_id - player_id) % 4] = 1

        self.curr_all_obs[player_id] = generate_obs(
            self.hand_tiles, self.river_tiles, self.side_tiles, self.dora_tiles,
            self.game_wind_obs, player_wind_obs, latest_tile=self.latest_tile)

        return self.curr_all_obs[player_id, :, :self.observation_space.shape[0]].swapaxes(0, 1)

    def get_full_obs(self, player_id):
        return self.curr_all_obs[player_id].swapaxes(0, 1)

    def get_state(self):
        # get raw state
        return self.hand_tiles, self.river_tiles, self.side_tiles, self.dora_tiles,\
            self.game_wind_obs, self.latest_tile

    def get_phase_text(self):
        return self.Phases[self.t.get_phase()]

    def step_play(self, action, playerNo):
        # self action phase
        current_playerNo = self.t.who_make_selection()

        if not self.t.get_phase() < 4:
            raise Exception("Current phase is not self-action phase!!")
        if not current_playerNo == playerNo:
            raise Exception("Current player is not the one neesd to execute action!!")

        info = {"playerNo": playerNo}
        score_before = self.t.players[playerNo].score

        aval_actions = self.t.get_self_actions()
        if aval_actions[action].action == mp.Action.Tsumo or aval_actions[action].action == mp.Action.KyuShuKyuHai:
            self.horas[playerNo] = True

        self.t.make_selection(action)

        if self.t.get_selected_action() == mp.Action.Play:
            self.played_a_tile[playerNo] = True

        new_state = self.get_state_(playerNo)

        if self.Phases[self.t.get_phase()] == "GAME_OVER":
            reward = self.get_final_score_change()[playerNo]
        else:
            reward = self.t.players[playerNo].score - score_before

        if self.Phases[self.t.get_phase()] == "GAME_OVER":
            done = 1
            self.final_score_changes.append(self.get_final_score_change())

        else:
            done = 0

        for i in range(4):
            self.scores_before[i] = self.t.players[i].score

        return new_state, reward, done, info

    def step_response(self, action:int, playerNo:int):
        # response phase

        current_playerNo = self.t.who_make_selection()

        if not self.t.get_phase() >= 4 and not self.t.get_phase == 16:
            raise Exception("Current phase is not response phase!!")
        if not current_playerNo == playerNo:
            raise Exception("Current player is not the one neesd to execute action!!")

        info = {"playerNo": playerNo}
        score_before = self.t.players[playerNo].score

        aval_actions = self.t.get_response_actions()

        if aval_actions[action].action == mp.Action.Ron or aval_actions[action].action == mp.Action.ChanKan or aval_actions[action].action == mp.Action.ChanAnKan:
            self.horas[playerNo] = True

        self.t.make_selection(action)

        self.played_a_tile[playerNo] = False
        new_state = self.get_state_(playerNo)

        if self.Phases[self.t.get_phase()] == "GAME_OVER":
            reward = self.get_final_score_change()[playerNo]
        else:
            reward = self.t.players[playerNo].score - score_before

        if self.Phases[self.t.get_phase()] == "GAME_OVER":
            done = 1
            self.final_score_changes.append(self.get_final_score_change())
        else:
            done = 0

        for i in range(4):
            self.scores_before[i] = self.t.players[i].score

        return new_state, reward, done, info

    def get_final_score_change(self):
        rewards = np.zeros([4], dtype=np.float32)

        for i in range(4):
            rewards[i] = self.t.get_result().score[i] - self.scores_before[i]

        return rewards

    def symmetric_matrix_features(self, matrix_features):
        """
        Generating a random alternative of features, which is symmetric to the original one
        !!!! Exception !!!! Green one color!!!
        :param hand_matrix_tensor: a B by 34 by 4 matrix, where B is batch_size
        :return: a new, symmetric matrix
        """
        perm_msp = np.random.permutation(3)

        matrix_features_new = np.zeros_like(matrix_features)

        ## msp
        tmp = []

        tmp.append(matrix_features[:, 0:9, :])
        tmp.append(matrix_features[:, 9:18, :])
        tmp.append(matrix_features[:, 18:27, :])

        matrix_features_new[:, 0:9, :] = tmp[perm_msp[0]]
        matrix_features_new[:, 9:18, :] = tmp[perm_msp[1]]
        matrix_features_new[:, 18:27, :] = tmp[perm_msp[2]]

        ## eswn
        tmp = []

        tmp.append(matrix_features[:, 27, :])
        tmp.append(matrix_features[:, 28, :])
        tmp.append(matrix_features[:, 29, :])
        tmp.append(matrix_features[:, 30, :])

        k = np.random.random_integers(0, 3)

        matrix_features_new[:, 27, :] = tmp[k % 4]
        matrix_features_new[:, 28, :] = tmp[(k+1) % 4]
        matrix_features_new[:, 29, :] = tmp[(k+2) % 4]
        matrix_features_new[:, 30, :] = tmp[(k+3) % 4]

        # chh
        tmp = []

        k = np.random.random_integers(0, 2)

        tmp.append(matrix_features[:, 31, :])
        tmp.append(matrix_features[:, 32, :])
        tmp.append(matrix_features[:, 33, :])

        matrix_features_new[:, 31, :] = tmp[k % 3]
        matrix_features_new[:, 32, :] = tmp[(k+1) % 3]
        matrix_features_new[:, 33, :] = tmp[(k+2) % 3]

        return matrix_features_new

    def get_aval_next_states(self, playerNo: int):

        current_playerNo = self.t.who_make_selection()
        if not current_playerNo == playerNo:
            raise Exception("Current player is not the one needs to execute action!!")

        S0 = self.t.get_next_aval_states_matrix_features_frost2(playerNo)
        s0 = self.t.get_next_aval_states_vector_features_frost2(playerNo)
        matrix_features = np.array(S0).reshape([-1, *self.matrix_feature_size])
        vector_features = np.array(s0).reshape([-1, self.vector_feature_size])

        return matrix_features, vector_features

    def tile_to_id(self, tile):
        # if tile == mp.BaseTile._1m:
        #     return 0
        # elif tile == mp.BaseTile._2m:
        #     return 1
        # elif tile == mp.BaseTile._3m:
        #     return 2
        # elif tile == mp.BaseTile._4m:
        #     return 3
        # elif tile == mp.BaseTile._5m:
        #     return 4
        # elif tile == mp.BaseTile._6m:
        #     return 5
        # elif tile == mp.BaseTile._7m:
        #     return 6
        # elif tile == mp.BaseTile._8m:
        #     return 7
        # elif tile == mp.BaseTile._9m:
        #     return 8
        # elif tile == mp.BaseTile._1p:
        #     return 9
        # elif tile == mp.BaseTile._2p:
        #     return 10
        # elif tile == mp.BaseTile._3p:
        #     return 11
        # elif tile == mp.BaseTile._4p:
        #     return 12
        # elif tile == mp.BaseTile._5p:
        #     return 13
        # elif tile == mp.BaseTile._6p:
        #     return 14
        # elif tile == mp.BaseTile._7p:
        #     return 15
        # elif tile == mp.BaseTile._8p:
        #     return 16
        # elif tile == mp.BaseTile._9p:
        #     return 17
        # elif tile == mp.BaseTile._1s:
        #     return 18
        # elif tile == mp.BaseTile._2s:
        #     return 19
        # elif tile == mp.BaseTile._3s:
        #     return 20
        # elif tile == mp.BaseTile._4s:
        #     return 21
        # elif tile == mp.BaseTile._5s:
        #     return 22
        # elif tile == mp.BaseTile._6s:
        #     return 23
        # elif tile == mp.BaseTile._7s:
        #     return 24
        # elif tile == mp.BaseTile._8s:
        #     return 25
        # elif tile == mp.BaseTile._9s:
        #     return 26
        # elif tile == mp.BaseTile.east:
        #     return 27
        # elif tile == mp.BaseTile.south:
        #     return 28
        # elif tile == mp.BaseTile.west:
        #     return 29
        # elif tile == mp.BaseTile.north:
        #     return 30
        # elif tile == mp.BaseTile.haku:
        #     return 31
        # elif tile == mp.BaseTile.hatsu:
        #     return 32
        # elif tile == mp.BaseTile.chu:
        #     return 33
        # else:
        #     raise Exception("Input must be a tile!!")
        return int(tile)

    def who_do_what(self):
        if self.t.get_phase() >= 4 and not self.t.get_phase() == 16:  # response phase
            return self.t.who_make_selection(), "response"
        elif self.t.get_phase() < 4:  # action phase
            return self.t.who_make_selection(), "play"

    def render(self, mode='human'):
        print(self.t.get_selected_action.action)




class EnvMahjong2(gym.Env):
    """
    Mahjong Environment for FrOst Ver2
    """

    metadata = {'name': 'Mahjong', 'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}
    spec = {'id': 'TaskT'}

    def __init__(self, printing=True):
        self.t = mp.Table()
        self.Phases = (
            "P1_ACTION", "P2_ACTION", "P3_ACTION", "P4_ACTION", "P1_RESPONSE", "P2_RESPONSE", "P3_RESPONSE",
            "P4_RESPONSE", "P1_抢杠RESPONSE", "P2_抢杠RESPONSE", "P3_抢杠RESPONSE", "P4_抢杠RESPONSE",
            "P1_抢暗杠RESPONSE", "P2_抢暗杠RESPONSE", " P3_抢暗杠RESPONSE", " P4_抢暗杠RESPONSE", "GAME_OVER",
            "P1_DRAW, P2_DRAW, P3_DRAW, P4_DRAW")
        self.horas = [False, False, False, False]
        self.played_a_tile = [False, False, False, False]
        self.tile_in_air = None
        self.final_score_changes = []
        self.game_count = 0
        self.printing = printing

        self.matrix_feature_size = [34, 58]
        self.vector_feature_size = 30

        self.scores_init = np.zeros([4], dtype=np.float32)
        for i in range(4):
            self.scores_init[i] = self.t.players[i].score

        self.scores_before = np.zeros([4], dtype=np.float32)
        for i in range(4):
            self.scores_before[i] = self.t.players[i].score

    def reset(self):
        self.t = mp.Table()

        oya = np.random.random_integers(0, 3)
        winds =['east', 'south', 'west', 'north']
        wind = winds[np.random.random_integers(0, 3)]
        oya = '{}'.format(oya)

        # self.t.game_init()
        self.t.game_init_with_metadata({"oya": oya, "wind": wind})

        self.game_count += 1

        self.horas = [False, False, False, False]
        self.played_a_tile = [False, False, False, False]

        self.scores_init = np.zeros([4], dtype=np.float32)
        for i in range(4):
            self.scores_init[i] = self.t.players[i].score

        self.scores_before = np.zeros([4], dtype=np.float32)
        for i in range(4):
            self.scores_before[i] = self.t.players[i].score

        return [self.get_state_(i) for i in range(4)]

    def get_phase_text(self):
        return self.Phases[self.t.get_phase()]

    # def step_draw(self, playerNo):
    #     current_playerNo = self.t.who_make_selection()
    #
    #     if not self.t.get_phase() < 4:
    #         raise Exception("Current phase is not draw phase!!")
    #     if not current_playerNo == playerNo:
    #         raise Exception("Current player is not the one neesd to execute action!!")
    #
    #     info = {"playerNo": playerNo}
    #     score_before = self.t.players[playerNo].score
    #
    #     self.played_a_tile[playerNo] = False
    #     new_state = self.get_state_(playerNo)
    #
    #     # the following should be unnecessary but OK
    #     if self.Phases[self.t.get_phase()] == "GAME_OVER":
    #         reward = self.get_final_score_change()[playerNo]
    #     else:
    #         reward = self.t.players[playerNo].score - score_before
    #
    #     if self.Phases[self.t.get_phase()] == "GAME_OVER":
    #         done = 1
    #         self.final_score_changes.append(self.get_final_score_change())
    #     else:
    #         done = 0
    #
    #     for i in range(4):
    #         self.scores_before[i] = self.t.players[i].score
    #
    #     return new_state, reward, done, info

    def step_play(self, action, playerNo):
        # self action phase
        current_playerNo = self.t.who_make_selection()

        if not self.t.get_phase() < 4:
            raise Exception("Current phase is not self-action phase!!")
        if not current_playerNo == playerNo:
            raise Exception("Current player is not the one neesd to execute action!!")

        info = {"playerNo": playerNo}
        score_before = self.t.players[playerNo].score

        aval_actions = self.t.get_self_actions()
        if aval_actions[action].action == mp.Action.Tsumo or aval_actions[action].action == mp.Action.KyuShuKyuHai:
            self.horas[playerNo] = True

        self.t.make_selection(action)

        if self.t.get_selected_action() == mp.Action.Play:
            self.played_a_tile[playerNo] = True

        new_state = self.get_state_(playerNo)

        if self.Phases[self.t.get_phase()] == "GAME_OVER":
            reward = self.get_final_score_change()[playerNo]
        else:
            reward = self.t.players[playerNo].score - score_before

        if self.Phases[self.t.get_phase()] == "GAME_OVER":
            done = 1
            self.final_score_changes.append(self.get_final_score_change())

        else:
            done = 0

        for i in range(4):
            self.scores_before[i] = self.t.players[i].score

        return new_state, reward, done, info

    def step_response(self, action:int, playerNo:int):
        # response phase

        current_playerNo = self.t.who_make_selection()

        if not self.t.get_phase() >= 4 and not self.t.get_phase == 16:
            raise Exception("Current phase is not response phase!!")
        if not current_playerNo == playerNo:
            raise Exception("Current player is not the one neesd to execute action!!")

        info = {"playerNo": playerNo}
        score_before = self.t.players[playerNo].score

        aval_actions = self.t.get_response_actions()

        if aval_actions[action].action == mp.Action.Ron or aval_actions[action].action == mp.Action.ChanKan or aval_actions[action].action == mp.Action.ChanAnKan:
            self.horas[playerNo] = True

        self.t.make_selection(action)

        self.played_a_tile[playerNo] = False
        new_state = self.get_state_(playerNo)

        if self.Phases[self.t.get_phase()] == "GAME_OVER":
            reward = self.get_final_score_change()[playerNo]
        else:
            reward = self.t.players[playerNo].score - score_before

        if self.Phases[self.t.get_phase()] == "GAME_OVER":
            done = 1
            self.final_score_changes.append(self.get_final_score_change())
        else:
            done = 0

        for i in range(4):
            self.scores_before[i] = self.t.players[i].score

        return new_state, reward, done, info

    def get_final_score_change(self):
        rewards = np.zeros([4], dtype=np.float32)

        for i in range(4):
            rewards[i] = self.t.get_result().score[i] - self.scores_before[i]

        return rewards

    def get_state_(self, playerNo):
        return 0

    def symmetric_matrix_features(self, matrix_features):
        """
        Generating a random alternative of features, which is symmetric to the original one
        !!!! Exception !!!! Green one color!!!
        :param hand_matrix_tensor: a B by 34 by 4 matrix, where B is batch_size
        :return: a new, symmetric matrix
        """
        perm_msp = np.random.permutation(3)

        matrix_features_new = np.zeros_like(matrix_features)

        ## msp
        tmp = []

        tmp.append(matrix_features[:, 0:9, :])
        tmp.append(matrix_features[:, 9:18, :])
        tmp.append(matrix_features[:, 18:27, :])

        matrix_features_new[:, 0:9, :] = tmp[perm_msp[0]]
        matrix_features_new[:, 9:18, :] = tmp[perm_msp[1]]
        matrix_features_new[:, 18:27, :] = tmp[perm_msp[2]]

        ## eswn
        tmp = []

        tmp.append(matrix_features[:, 27, :])
        tmp.append(matrix_features[:, 28, :])
        tmp.append(matrix_features[:, 29, :])
        tmp.append(matrix_features[:, 30, :])

        k = np.random.random_integers(0, 3)

        matrix_features_new[:, 27, :] = tmp[k % 4]
        matrix_features_new[:, 28, :] = tmp[(k+1) % 4]
        matrix_features_new[:, 29, :] = tmp[(k+2) % 4]
        matrix_features_new[:, 30, :] = tmp[(k+3) % 4]

        # chh
        tmp = []

        k = np.random.random_integers(0, 2)

        tmp.append(matrix_features[:, 31, :])
        tmp.append(matrix_features[:, 32, :])
        tmp.append(matrix_features[:, 33, :])

        matrix_features_new[:, 31, :] = tmp[k % 3]
        matrix_features_new[:, 32, :] = tmp[(k+1) % 3]
        matrix_features_new[:, 33, :] = tmp[(k+2) % 3]

        return matrix_features_new

    def get_aval_next_states(self, playerNo: int):

        current_playerNo = self.t.who_make_selection()
        if not current_playerNo == playerNo:
            raise Exception("Current player is not the one needs to execute action!!")

        S0 = self.t.get_next_aval_states_matrix_features_frost2(playerNo)
        s0 = self.t.get_next_aval_states_vector_features_frost2(playerNo)
        matrix_features = np.array(S0).reshape([-1, *self.matrix_feature_size])
        vector_features = np.array(s0).reshape([-1, self.vector_feature_size])

        return matrix_features, vector_features

    def tile_to_id(self, tile):
        # if tile == mp.BaseTile._1m:
        #     return 0
        # elif tile == mp.BaseTile._2m:
        #     return 1
        # elif tile == mp.BaseTile._3m:
        #     return 2
        # elif tile == mp.BaseTile._4m:
        #     return 3
        # elif tile == mp.BaseTile._5m:
        #     return 4
        # elif tile == mp.BaseTile._6m:
        #     return 5
        # elif tile == mp.BaseTile._7m:
        #     return 6
        # elif tile == mp.BaseTile._8m:
        #     return 7
        # elif tile == mp.BaseTile._9m:
        #     return 8
        # elif tile == mp.BaseTile._1p:
        #     return 9
        # elif tile == mp.BaseTile._2p:
        #     return 10
        # elif tile == mp.BaseTile._3p:
        #     return 11
        # elif tile == mp.BaseTile._4p:
        #     return 12
        # elif tile == mp.BaseTile._5p:
        #     return 13
        # elif tile == mp.BaseTile._6p:
        #     return 14
        # elif tile == mp.BaseTile._7p:
        #     return 15
        # elif tile == mp.BaseTile._8p:
        #     return 16
        # elif tile == mp.BaseTile._9p:
        #     return 17
        # elif tile == mp.BaseTile._1s:
        #     return 18
        # elif tile == mp.BaseTile._2s:
        #     return 19
        # elif tile == mp.BaseTile._3s:
        #     return 20
        # elif tile == mp.BaseTile._4s:
        #     return 21
        # elif tile == mp.BaseTile._5s:
        #     return 22
        # elif tile == mp.BaseTile._6s:
        #     return 23
        # elif tile == mp.BaseTile._7s:
        #     return 24
        # elif tile == mp.BaseTile._8s:
        #     return 25
        # elif tile == mp.BaseTile._9s:
        #     return 26
        # elif tile == mp.BaseTile.east:
        #     return 27
        # elif tile == mp.BaseTile.south:
        #     return 28
        # elif tile == mp.BaseTile.west:
        #     return 29
        # elif tile == mp.BaseTile.north:
        #     return 30
        # elif tile == mp.BaseTile.haku:
        #     return 31
        # elif tile == mp.BaseTile.hatsu:
        #     return 32
        # elif tile == mp.BaseTile.chu:
        #     return 33
        # else:
        #     raise Exception("Input must be a tile!!")
        return int(tile)

    def dora_ind_2_dora_id(self, ind_id):
        if ind_id == 8:
            return 0
        elif ind_id == 8 + 9:
            return 0 + 9
        elif ind_id == 8 + 9 + 9:
            return 0 + 9 + 9
        elif ind_id == 33:
            return 31
        else:
            return ind_id + 1

    def who_do_what(self):
        if self.t.get_phase() >= 4 and not self.t.get_phase() == 16:  # response phase
            return self.t.who_make_selection(), "response"
        elif self.t.get_phase() < 4:  # action phase
            return self.t.who_make_selection(), "play"

    def render(self, mode='human'):
        print(self.t.get_selected_action.action)


