import re
import os
import warnings

import numpy as np
import xml.etree.ElementTree as ET
import urllib.request
import gzip


def generate_obs(hand_tiles, river_tiles, side_tiles, dora_tiles, game_wind, self_wind):

    this_player_obs = np.zeros([34, 40], dtype=np.uint8)
    this_oracle_obs = np.zeros([34, 15], dtype=np.uint8)

    # ----------------- River Tiles Procces ------------------
    for player_river_tile in river_tiles:
        pass

    # ----------------- Side Tiles Process ------------------
    for player_side_tile in side_tiles:
        pass

    # ----------------- Hand Tiles Process ------------------

    for player_hand_tile in hand_tiles:
        pass

    # ----------------- Dora Process ------------------
    for dora_tile in dora_tiles:
        dora_hai_id = int(dora_tile / 4)
        this_player_obs[dora_hai_id, - 5] += 1

    return this_player_obs, this_oracle_obs


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

            game_wind = np.zeros(34)  # index: -4
            game_wind[27] = 1

            self_wind = np.zeros([4, 34])  # index: -3
            self_wind[0, 27 + (4 - oya_id) % 4] = 1
            self_wind[1, 27 + (5 - oya_id) % 4] = 1
            self_wind[2, 27 + (6 - oya_id) % 4] = 1
            self_wind[3, 27 + (7 - oya_id) % 4] = 1

            dora_tiles = [int(child.get("seed").split(",")[-1])]

            hand_tiles = []
            for player_id in range(4):
                tiles_str = child.get("hai{}".format(player_id)).split(",")
                hand_tiles_player = [int(tmp) for tmp in tiles_str]
                hand_tiles.append(hand_tiles_player)

            river_tiles = [[], [], [], []]  # each has two elements: tile_no and is_from_hand
            side_tiles = [[], [], [], []]  # each has 4 elements: tiles_no and naru_tile (e.g., [4, 8, 12, 8])

            # ----------------------- Generate initial player and oracle observaTION  ----------------

            curr_player_obs, curr_oracle_obs = generate_obs(
                hand_tiles, river_tiles, side_tiles, dora_tiles, game_wind, self_wind)

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
                curr_player_obs[new_dora_hai_id, -5] += 1

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
                #                 print("掉线！！！！！！！！！！")
                pass
                # record_this_game = False
            elif child.tag == "RYUUKYOKU" or child.tag == "AGARI":
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

                # if np.any(np.array(scores_change[0::2]) + np.array(scores_change[1::2]) < 0):
                #     print(scores_change)
                # num_bars_ = child.get("ba").split(",")
                # num_bars = int(num_bars_[1])
                # if np.sum(rewards) + num_bars * 10 != 0:
                #     print(child.get("sc"))
                ## If someone was flied, reward must be modified because the
                #                 print(child.tag, child.attrib)
                #                 print("results:", rewards)

                for player_id in range(4):
                    sum_scores[player_id] += rewards[player_id]
                    scores_change_this_game[player_id] += rewards[player_id]

                # oya_scores += rewards[oya]

                if "owari" in child.attrib:
                    owari_scores_change_str = child.get("sc").split(",")
                    owari_scores_change = [int(tmp) for tmp in owari_scores_change_str]
                    if np.sum(owari_scores_change) > 1000:
                        print(owari_scores_change)

                    # for player_id in range(4):
                    #     sum_scores[player_id] += owari_scores_change[player_id * 2] + owari_scores_change[player_id * 2 + 1] - 250

                # if child.tag == "RYUUKYOKU":
                #     print("------------------")
                #     print(child.get("ba"))
                #     print(scores_change_this_game)
                #     print("------------------")

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

