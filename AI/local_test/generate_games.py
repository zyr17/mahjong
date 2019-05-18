from naiveAI import AgentNaive, NMnaive
import tensorflow as tf
import numpy as np
from copy import deepcopy
import MahjongPy as mp
import argparse, os, warnings

parser = argparse.ArgumentParser()
parser.add_argument('id', type=int, help="array ID")
args = parser.parse_args()

id = args.id


savepath = '../data190501/'

filename = 'Experiences' + '-%4.4d' % (id)

if os.path.exists(savepath):
    warnings.warn('{} exists (possibly so do data).'.format(savepath))
else:
    os.makedirs(savepath)

Phases = \
("P1_ACTION", "P2_ACTION", "P3_ACTION", "P4_ACTION", "P1_RESPONSE", "P2_RESPONSE", "P3_RESPONSE", "P4_RESPONSE",
"P1_抢杠RESPONSE", "P2_抢杠RESPONSE", "P3_抢杠RESPONSE", "P4_抢杠RESPONSE",
"P1_抢暗杠RESPONSE", "P2_抢暗杠RESPONSE", " P3_抢暗杠RESPONSE", " P4_抢暗杠RESPONSE", "GAME_OVER")



required_games = 1024

count = 0

while count < required_games:
    t = mp.Table()
    t.game_init()
    for m in range(500):
        # print(Phases[t.get_phase()] )

        if t.get_phase() < 4:
            aval_actions = t.get_self_actions()
            a = np.random.randint(len(aval_actions))
            if aval_actions[a].action == mp.Action.Tsumo:
                print(aval_actions[a].action)
                for i in range(len(aval_actions[a].correspond_tiles)):
                    print(aval_actions[a].correspond_tiles[i].tile)
            t.make_selection(a)
        else:
            aval_actions = t.get_response_actions()
            a = np.random.randint(len(aval_actions))

            if aval_actions[a].action == mp.Action.Ron:
                print(aval_actions[a].action)
                for i in range(len(aval_actions[a].correspond_tiles)):
                    print(aval_actions[a].correspond_tiles[i].tile)
            t.make_selection(a)

        # print("\r")

        if Phases[t.get_phase()] == "GAME_OVER":
            print(t.get_result().result_type)
            break

    if t.get_result().result_type == mp.ResultType.RonAgari or t.get_result().result_type == mp.ResultType.TsumoAgari:
        break


