import re
import os
import warnings

import numpy as np
import xml.etree.ElementTree as ET
import urllib.request
import gzip


# ======================== Terminology explain ======================

# Match: A set of games on one table until someone flys away or a series of games is finished
# Game: from Init to Agari or Ryukyouku
# Episode: 1 game for 1 player

player_i_hand_start_ind = [0, 63, 69, 75]  # later 3 in oracle_obs
player_i_side_start_ind = [6, 12, 18, 24]
player_i_river_start_ind = [30, 37, 44, 51]

dora_indicator_ind = 58
dora_ind = 59
game_wind_ind = 60
self_wind_ind = 61
wait_tile_ind = 62

aka_tile_ids = [16, 16 + 36, 16 + 36 + 36]

player_obs_width = 63


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


def generate_obs(hand_tiles, river_tiles, side_tiles, dora_tiles, game_wind, self_wind):

    all_obs = np.zeros([4, 34, 63 + 18], dtype=np.uint8)

    global player_i_hand_start_ind
    global player_i_side_start_ind
    global player_i_river_start_ind

    global dora_indicator_ind
    global dora_ind
    global game_wind_ind
    global self_wind_ind
    global wait_tile_ind

    global aka_tile_ids

    # ----------------- Side Tiles Process ------------------
    for player_id, player_side_tiles in enumerate(side_tiles):
        side_tile_num = np.zeros(34, dtype=np.uint8)
        for side_tile in player_side_tiles:
            side_tile_id = int(side_tile[0] / 4)
            side_tile_num[side_tile_id] += 1

            if side_tile[0] in aka_tile_ids:
                # Red dora
                all_obs[player_id, side_tile_id, player_i_side_start_ind[player_id] + 5] = 1

            if side_tile[1] == 1:
                # Naru tile
                all_obs[player_id, side_tile_id, player_i_side_start_ind[player_id] + 4] = 1

        for t_id in range(34):
            for k in range(4):
                if side_tile_num[t_id] > k:
                    all_obs[player_id, t_id, player_i_side_start_ind[player_id] + k] = 1

    # ----------------- River Tiles Procces ------------------
    for player_id, player_river_tiles in enumerate(river_tiles):  # 副露也算在牌河里, also include Riichi info
        river_tile_num = np.zeros(34, dtype=np.uint8)
        for river_tile in player_river_tiles:
            river_tile_id = int(river_tile[0] / 4)
            river_tile_num[river_tile_id] += 1

            if river_tile[0] in aka_tile_ids:
                # Red dora
                all_obs[player_id, river_tile_id, player_i_river_start_ind[player_id] + 5] = 1

            if river_tile[1] == 1:
                # te-kiri (from hand)
                all_obs[player_id, river_tile_id, player_i_river_start_ind[player_id] + 4] += 1

            if river_tile[2] == 1:
                # is riichi-announcement tile
                all_obs[player_id, river_tile_id, player_i_river_start_ind[player_id] + 6] = 1

        for t_id in range(34):
            for k in range(4):
                if river_tile_num[t_id] > k:
                    all_obs[player_id, t_id, player_i_river_start_ind[player_id] + k] = 1

    # ----------------- Hand Tiles Process ------------------
    for player_id, player_hand_tiles in enumerate(hand_tiles):
        hand_tile_num = np.zeros(34, dtype=np.uint8)
        for hand_tile in player_hand_tiles:
            hand_tile_id = int(hand_tile / 4)
            hand_tile_num[hand_tile_id] += 1

            if hand_tile in aka_tile_ids:
                # Aka dora
                all_obs[player_id, hand_tile_id, player_i_hand_start_ind[player_id] + 5] = 1

            # how many times this tile has been discarded before by this player
            all_obs[player_id, hand_tile_id, player_i_hand_start_ind[player_id] + 4] = np.sum(
                all_obs[player_id, hand_tile_id, player_i_river_start_ind[player_id]:player_i_river_start_ind[player_id] + 4])

        for t_id in range(34):
            for k in range(4):
                if hand_tile_num[t_id] > k:
                    all_obs[player_id, t_id, player_i_hand_start_ind[player_id] + k] = 1

    # ----------------- Dora Process ------------------
    for dora_tile in dora_tiles:
        dora_hai_id = int(dora_tile / 4)
        all_obs[:, dora_hai_id, dora_ind] += 1
        all_obs[:, dora2indicator(dora_hai_id), dora_indicator_ind] += 1

    # ----------------- Public Game State ----------------
    all_obs[:, :, game_wind_ind] = game_wind  # Case 1 to 4 in dim 0
    all_obs[:, :, self_wind_ind] = self_wind

    players_obs = all_obs[:, :, :63]
    oracles_obs = all_obs[:, :, 63:]

    return players_obs, oracles_obs


paipu_urls = []

path = "../2020_paipu"
files = os.listdir(path)  # 得到文件夹下的所有文件名称

filenames = []

for file in files:  # 遍历文件夹
    if not os.path.isdir(file) and file[-4:] == "html":  # 判断是否是文件夹，不是文件夹才打开
        filenames.append(path + "/" + file)  # 打开文件
        print(file)

for filename in filenames:

    f = open(filename, 'r', encoding='UTF-8')

    scc = f.read()
    # print(scc)

    f.close()

    replay_urls = re.findall('href="(.+?)">', scc)

    log_urls = []

    for tmp in replay_urls:
        log_url_split = tmp.split("?log=")
        log_urls.append(log_url_split[0] + "log?" + log_url_split[1])

    paipu_urls = paipu_urls + log_urls

# -------------- Hyper-parameters ------------------

max_ten_diff = 250  # 最大点数限制，排除点数差距过大时的非正常打法
min_dan = 15  # 最低段位限制，可以排除三麻的局（三麻的缺省player的dan=0）

max_aval_action_num = 16
max_all_steps = 100000
max_steps = 200

player_obs_total = np.zeros([max_all_steps, 1, 34, 40], dtype=np.uint8)
oracle_obs_total = np.zeros([max_all_steps, 1, 34, 15], dtype=np.uint8)
player_actions_total = np.zeros([max_all_steps, max_aval_action_num, 1, 34, 40], dtype=np.uint8)
oracle_actions_total = np.zeros([max_all_steps, max_aval_action_num, 1, 34, 15], dtype=np.uint8)
aval_actions_num_total = np.zeros([max_all_steps], dtype=np.uint8)

done_total = np.zeros([max_all_steps], dtype=np.float32)
reward_total = np.zeros([max_all_steps], dtype=np.float32)

hosts = ["e3.mjv.jp",
         "e4.mjv.jp",
         "e5.mjv.jp",
         "k0.mjv.jp",
         "e.mjv.jp"]

num_games = 0
game_has_init = False

sum_scores = np.zeros(4, dtype=np.float64)
oya_scores = np.zeros(1, dtype=np.float64)

machi_hai_freq = np.zeros(136, dtype=np.float64)
# ----------------- start ---------------------

for url in paipu_urls:
    for host in hosts:

        try:
            HEADER = {
                'Host': host,
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:65.0) Gecko/20100101 Firefox/65.0',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive'
            }

            req = urllib.request.Request(url=url, headers=HEADER)
            opener = urllib.request.build_opener()
            response = opener.open(req)
            #         print(response.read())
            paipu = gzip.decompress(response.read()).decode('utf-8')
            #         print(paipu)
            break
        except:
            pass

    root = ET.fromstring(paipu)

    # =================== 开始解析牌谱 =======================
    record_this_game = True

    for child_no, child in enumerate(root):
        # Initial information, discard
        if child.tag == "SHUFFLE":
            #         print(child.attrib)
            pass

        elif child.tag == "GO":  # 牌桌规则和等级等信息.
            #         print(child.attrib)
            type_num = int(child.get("type"))
            tmp = str(bin(type_num))

            game_info = dict()
            game_info["is_pvp"] = int(tmp[-1])
            game_info["no_aka"] = int(tmp[-2])
            game_info["no_kuutan"] = int(tmp[-3])
            game_info["is_hansou"] = int(tmp[-4])
            game_info["is_3ma"] = int(tmp[-5])
            #             if game_info["is_3ma"]:
            #                 print("!!!!!!! Is 3 Ma !!!!!!!!!!!!!!!!")
            game_info["is_pro"] = int(tmp[-6])
            game_info["is_fast"] = int(tmp[-7])
            game_info["is_joukyu"] = int(tmp[-8])

            # 0x01	如果是PVP对战则为1
            # 0x02	如果没有赤宝牌则为1
            # 0x04	如果无食断则为1
            # 0x08	如果是半庄则为1
            # 0x10	如果是三人麻将则为1
            # 0x20	如果是特上卓或凤凰卓则为1
            # 0x40	如果是速卓则为1
            # 0x80	如果是上级卓则为1

        elif child.tag == "TAIKYOKU":
            pass

        elif child.tag == "UN":
            #         print(child.attrib)
            if "dan" in child.attrib:
                dans_str = child.get("dan").split(',')
                dans = [int(tmp) for tmp in dans_str]

                if min(dans) < min_dan:
                    break  # not record this whole game

        elif child.tag == "INIT":
            record_this_game = True

            scores_change_this_game = np.zeros([4])
            #         print(child.attrib)

            #         print("------------------------------------")
            # player_obs = np.zeros([4, max_steps, 1, 34, 60], dtype=np.uint8)
            # oracle_obs = np.zeros([4, max_steps, 1, 34, 15], dtype=np.uint8)
            # player_actions = np.zeros([4, max_steps, max_aval_action_num, 1, 34, 60], dtype=np.uint8)
            # oracle_actions = np.zeros([4, max_steps, max_aval_action_num, 1, 34, 15], dtype=np.uint8)
            # aval_actions_num = np.zeros([4, max_steps], dtype=np.uint8)

            # scores_number
            scores_str = child.get("ten").split(',')
            scores = [int(tmp) for tmp in scores_str]

            #             if max(scores) - min(scores) > max_ten_diff:
            #                 record_this_game = False
            #             else:
            #                 record_this_game = True
            #         print(scores)

            # Oya number
            oya_id = int(child.get("oya"))

            game_wind_obs = np.zeros(34)  # index: -4
            game_wind_obs[27] = 1

            self_wind_obs = np.zeros([4, 34])  # index: -3
            self_wind_obs[0, 27 + (4 - oya_id) % 4] = 1
            self_wind_obs[1, 27 + (5 - oya_id) % 4] = 1
            self_wind_obs[2, 27 + (6 - oya_id) % 4] = 1
            self_wind_obs[3, 27 + (7 - oya_id) % 4] = 1

            dora_tiles = [int(child.get("seed").split(",")[-1])]

            hand_tiles = []
            for player_id in range(4):
                tiles_str = child.get("hai{}".format(player_id)).split(",")
                hand_tiles_player = [int(tmp) for tmp in tiles_str]
                hand_tiles.append(hand_tiles_player)

            river_tiles = [[], [], [], []]  # each has 3 elements: tile_no, is_from_hand and riichi_announce_tile
            side_tiles = [[], []]  # each has 2 elements: tile_no and is_naru_tile

            # ----------------------- Generate initial player and oracle observaTION  ----------------

            curr_players_obs, curr_oracles_obs = generate_obs(
                hand_tiles, river_tiles, side_tiles, dora_tiles, game_wind_obs, self_wind_obs)

            game_has_init = True

        # ------------------------- Actions ---------------------------

        elif record_this_game:

            if not game_has_init:
                record_this_game = False
                for cc in range(min(child_no + 4, len(root))):
                    print(root[cc].tag, root[cc].attrib)
                warnings.warn("============= Game has not been correctly initialized, skipped ================")
                continue

            if child.tag == "DORA":
                dora_tiles.append(int(child.get("hai")))
                new_dora_hai_id = int(int(child.get("hai")) / 4)

                curr_player_obs
                curr_player_obs[new_dora_hai_id, -4] += 1
                new_dora_indicator_id = dora2indicator(new_dora_hai_id)
                curr_player_obs[new_dora_indicator_id, -5] += 1

            elif child.tag == "REACH":
                if int(child.get("step")) == 2:
                    player_id = int(child.get("who"))
                    sum_scores[player_id] -= 10
                    scores_change_this_game[player_id] -= 10
                    # if oya == player_id:
                    #     oya_scores -= 10
            elif child.tag[0] in ["T", "U", "V", "W"]:  # 摸牌
                pass
            elif child.tag[0] in ["D", "E", "F", "G"]:  # 打牌
                pass
            elif child.tag == "N":  # 鸣牌
                pass
            elif child.tag == "BYE":  # 掉线
                record_this_game = False
                continue
            elif child.tag == "RYUUKYOKU" or child.tag == "AGARI":

                # ------------------- Statistics -------------------------

                scores_change_str = child.get("sc").split(",")
                scores_change = [int(tmp) for tmp in scores_change_str]
                rewards = scores_change[1::2]

                oya_scores += rewards[oya_id]

                if child.tag == "AGARI":

                    # double-ron
                    if len(child.get("who")) > 1:
                        for c in root:
                            print(c.tag, c.attrib)
                        raise ValueError("from who is not single player!!!")

                    machi_hai_str = child.get("machi").split(",")
                    machi_hai = np.array([int(tmp) for tmp in machi_hai_str]).astype(np.int)

                    machi_hai_freq[machi_hai] += 1

                for player_id in range(4):
                    sum_scores[player_id] += rewards[player_id]
                    scores_change_this_game[player_id] += rewards[player_id]

                if "owari" in child.attrib:
                    owari_scores_change_str = child.get("sc").split(",")
                    owari_scores_change = [int(tmp) for tmp in owari_scores_change_str]
                    if np.sum(owari_scores_change) > 1000:
                        print(owari_scores_change)

                num_games += 1
                if num_games % 100 == 0:
                    print(num_games)
                    print("avg_scores:", sum_scores / num_games)
                    print("avg_oya_scores:", oya_scores / num_games)
                    # print("machi hai frequency:", machi_hai_freq / num_games)

                if child_no + 1 < len(root) and root[child_no + 1].tag == "AGARI":
                    game_has_init = True  # many players agari
                else:
                    game_has_init = False

            else:
                print(child.tag, child.attrib)

                raise ValueError("Unexpected Element!")

