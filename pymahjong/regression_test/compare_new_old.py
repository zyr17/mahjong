import time
import numpy as np
import torch
import MahjongPyWrapper as pm
import MahjongPy as pm_old
import env_mahjong
import argparse
import scipy.io as sio

def play_mahjong(agent, num_games=100, verbose=1, type="new", step_pause=0):

    np.random.seed(0)

    env_new = env_mahjong.EnvMahjong4(type=pm.Table)
    env_old = env_mahjong.EnvMahjong4(type=pm_old.Table)

    start_time = time.time()
    game = 0
    success_games = 0

    stat = {}
    # stat["agari_games"] = np.zeros([4], dtype=np.float32)
    # stat["tsumo_games"] = np.zeros([4], dtype=np.float32)
    # stat["agari_points"] = np.zeros([4], dtype=np.float32)
    # stat["houjyuu_games"] = np.zeros([4], dtype=np.float32)

    while game < num_games:

        # try:

        env_new.reset(oya=game % 4, game_wind="east")
        env_old.reset(oya=game % 4, game_wind="east")

        step = 0

        executor_obs_episode_new = np.zeros([1000, 93, 34])
        executor_obs_episode_old = np.zeros([1000, 93, 34])

        while not (env_new.is_over() or env_old.is_over()):

            curr_player_id_new = env_new.get_curr_player_id()
            curr_player_id_old = env_old.get_curr_player_id()
            # --------- get decision information -------------

            valid_actions_mask_new = env_new.get_valid_actions(nhot=True)
            executor_obs_new = env_new.get_obs(curr_player_id_new)

            valid_actions_mask_old = env_old.get_valid_actions(nhot=True)
            executor_obs_old = env_old.get_obs(curr_player_id_old)

            oracle_obs_new = env_new.get_oracle_obs(curr_player_id_new)
            oracle_obs_old = env_old.get_oracle_obs(curr_player_id_old)
            # full_obs = env.get_full_obs(curr_player_id)
            # full_obs = concat([executor_obs, oracle_obs], axis=0)

            executor_obs_episode_new[step] = executor_obs_new
            executor_obs_episode_old[step] = executor_obs_old

            # --------- make decision -------------

            a_new = agent.select(executor_obs_new, oracle_obs_new, action_mask=valid_actions_mask_new, greedy=True)
            a_old = agent.select(executor_obs_old, oracle_obs_old, action_mask=valid_actions_mask_old, greedy=True)

            if np.any(valid_actions_mask_new != valid_actions_mask_old) or (curr_player_id_new != curr_player_id_old):
                print("======== New one======= game {} step {} === before making decision, phase {} ============".format(game, step, env_new.get_phase_text()))
                print("aval_actions_len = ", np.count_nonzero(valid_actions_mask_new))
                print(np.argwhere(valid_actions_mask_new).flatten())
                print("******************  agent select", a_new)
                if step > 0:
                    env_new.render()

                print("======== Old one======= game {} step {} === before making decision, phase {} ============".format(game, step, env_old.get_phase_text()))
                print("aval_actions_len = ", np.count_nonzero(valid_actions_mask_old))
                print(np.argwhere(valid_actions_mask_old).flatten())
                print("******************  agent select", a_old)
                if step > 0:
                    env_old.render()

                break

            # if np.any(executor_obs_new != executor_obs_old):
            #     print("different obs:", np.argwhere(executor_obs_new - executor_obs_old))

            env_new.step(curr_player_id_new, a_new)
            env_old.step(curr_player_id_old, a_old)
            step += 1

            time.sleep(step_pause)

        # ----------------------- get result ---------------------------------

        # sio.savemat("game{}_type-{}.mat".format(game, type), {"executor_obs": executor_obs_episode[:step]})
        #
        # payoffs = np.array(env.get_payoffs())
        # if verbose >= 2:
        #     print("Game {}, result: {}".format(game, payoffs))
        #     env.render()
        #
        # if env.t.get_result().result_type == pm.ResultType.RonAgari:
        #     winner = np.argwhere(payoffs > 0).flatten()
        #     loser = np.argmin(payoffs)
        #     stat["agari_points"][winner] += payoffs[winner]
        #     stat["agari_games"][winner] += 1
        #     stat["houjyuu_games"][loser] += 1
        #
        #     print("--------------player {} 放炮给 player {}-----------------".format(loser, winner))
        #     print(env.t.get_result().to_string())
        #     env.render()
        #
        # if env.t.get_result().result_type == pm.ResultType.TsumoAgari:
        #     winner = np.argmax(payoffs)
        #     stat["agari_points"][winner] += payoffs[winner]
        #     stat["agari_games"][winner] += 1
        #     stat["tsumo_games"][winner] += 1

        success_games += 1
        game += 1

        # if verbose >= 1 and game % 10 == 1:
        #     print("------------------------ {} games statistics -----------------------".format(success_games))
        #     print("win rate:                    ", np.array2string(100 * stat["agari_games"] / success_games, precision=2, separator="  "))
        #     print("tsumo rate:                  ", np.array2string(100 * stat["tsumo_games"] / stat["agari_games"], precision=2, separator="  "))
        #     print("houjyuu rate:                ", np.array2string(100 * stat["houjyuu_games"] / success_games, precision=2, separator="  "))
        #     print("average agari points (x100): ", np.array2string(0.01 * stat["agari_points"] / stat["agari_games"], precision=2, separator="  "))
        #     # print("----------------------------------------------------------------------")
        # except:
        #     game += 1
        #     continue

    print("Total {} game, {} without error, takes {} s".format(num_games, success_games, time.time() - start_time))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, help="use new or old C++ Mahjong code")

    args = parser.parse_args()

    torch.manual_seed(4321)
    # torch.seed()

    agent = torch.load("./mahjong_VLOG_CQL_0.model", map_location='cpu')
    agent.device = torch.device('cpu')

    # agent = "random"

    play_mahjong(agent, num_games=100, verbose=1, type=args.type, step_pause=0)
