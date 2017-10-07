import os
import sys
import datetime
import time
import shutil
import multiprocessing
import torch
import json
import random
import traceback
import numpy as np
from battle_client import load_model, play
from generate_tensor import extend_all_chessman, get_chessman_state_index, regular_chessboard
from matrix2tensor import matrix2tensor
from multiway_tree import add_node, write_tree, merge_tree
from utility import show_channels

from collect_tensor_and_label import collect_tensor_and_label
from train import train


def init_players():
    players = {}
    for player_id in range(1, 5):
        add_node(players, [player_id, "player_id"], player_id)
        for chessman_level_i in range(5):
            add_node(players, [player_id, "chessman_num_on_board", str(chessman_level_i + 1), 0])
        add_node(players, [player_id, "bonus_all_clear"], 0)
        add_node(players, [player_id, "bonus_101_is_last"], 0)
        add_node(players, [player_id, "score"], 0)
    return players


def battle_itself(model_offensive, model_defensive, tensor_output_folder,
                  squareness_output_file, is_use_cuda, is_show_tensor):
    chessman_dic = extend_all_chessman()
    chessman_state_id, chessman_state_id_inverse = get_chessman_state_index(chessman_dic)
    is_finish = np.array([False, False, False, False])
    channel_size = 20

    chessboard_player_id = np.zeros((channel_size, channel_size))
    chessboard_chessman_id = np.zeros((channel_size, channel_size))
    chessboard_hand_no = np.zeros((channel_size, channel_size))

    all_action = []
    game_state = {
        "msg_name": "game_start",
        "msg_data": {
            "players": [
                {"team_id": 1, "team_name": "tt", "player_id": 1, "birthplace": {"x": 0, "y": 0}},
                {"team_id": 0, "team_name": "cc", "player_id": 2, "birthplace": {"x": 0, "y": 19}},
                {"team_id": 1, "team_name": "tt", "player_id": 3, "birthplace": {"x": 19, "y": 10}},
                {"team_id": 0, "team_name": "cc", "player_id": 4, "birthplace": {"x": 19, "y": 0}},
            ]
        },
        "time": int(time.time())
    }
    all_action.append(json.dumps(game_state))

    label = {}
    players = init_players()
    player_id = 0
    hand_no = 0
    hand_no_threshold = random.randint(4, 20)
    while not is_finish.all():
        player_id += 1
        player_id %= 4
        player_id = player_id if player_id != 0 else 4

        if is_finish[player_id - 1]:
            continue

        hand_no += 1
        model = model_defensive
        if player_id == 1 or player_id == 3:
            model = model_offensive

        chessboard_player_id_regulated = regular_chessboard(chessboard_player_id, player_id,
                                                            is_replace_player_id=True)
        chessboard_chessman_id_regulated = regular_chessboard(chessboard_chessman_id, player_id)
        chessboard_hand_no_regulated = regular_chessboard(chessboard_hand_no, player_id)
        tensor = matrix2tensor(chessboard_player_id_regulated,
                               chessboard_chessman_id_regulated,
                               chessboard_hand_no_regulated)
        show_channels(tensor, is_show_tensor, True)
        if hand_no <= hand_no_threshold:
            maxk = 10
            random_choose_rate = 0.9
        else:
            maxk = 5
            random_choose_rate = 0.1

        chessman, chessman_id, state, x, y = play(chessman_dic, chessman_state_id_inverse, tensor, model, player_id,
                                                  is_use_cuda, maxk, random_choose_rate)
        if not chessman:
            is_finish[player_id - 1] = True
            my_team_id = player_id % 2
            action = {
                "msg_name": "notification",
                "msg_data": {
                    "hand": {
                        "no": hand_no
                    },
                    "team_id": my_team_id,
                    "player_id": player_id,
                    "chessman": {
                        "cause": "end"
                    }
                },
                "time": int(time.time())
            }
            all_action.append(json.dumps(action))

            # add_node(players, [player_id, "player_id"], player_id)
            if player_id not in players or "chessman_num_on_board" not in players[player_id]:
                for chessman_level_i in range(5):
                    add_node(players, [player_id, "chessman_num_on_board", str(chessman_level_i + 1)], 0)
            if sum(players[player_id]["chessman_num_on_board"].values()) == 21:
                add_node(players, [player_id, "bonus_all_clear"], 15)
                add_node(players, [player_id, "score"], 15)
            else:
                add_node(players, [player_id, "bonus_all_clear"], 0)
                add_node(players, [player_id, "bonus_101_is_last"], 0)
            continue

        if hand_no > hand_no_threshold:
            tensor_save_file = os.path.join(tensor_output_folder, str(hand_no) + '.npy')
            np.save(tensor_save_file, tensor)
            add_node(label, [player_id, tensor_save_file],
                     [chessman_id, state, x, y,
                      channel_size * y + x +
                      channel_size * channel_size * chessman_state_id[chessman_id][state]])
        chessman_level = int(chessman_id / 100)
        add_node(players, [player_id, "chessman_num_on_board", str(chessman_level)], 1)
        add_node(players, [player_id, "score"], chessman_level)
        if chessman_id == 10:
            if sum(players[player_id]["chessman_num_on_board"].values()) == 20:
                add_node(players, [player_id, "bonus_101_is_last"], 5)
                add_node(players, [player_id, "score"], 5)

        squareness = chessman["squareness"]
        for grid in squareness:
            chessboard_player_id[grid['x']][grid['y']] = player_id
            chessboard_chessman_id[grid['x']][grid['y']] = chessman_id
            chessboard_hand_no[grid['x']][grid['y']] = hand_no

        my_team_id = player_id % 2

        action = {
            "msg_name": "notification",
            "msg_data": {
                "hand": {
                    "display": squareness[0],
                    "no": hand_no
                },
                "team_id": my_team_id,
                "player_id": player_id,
                "chessman": chessman
            },
            "time": int(time.time())
        }

        all_action.append(json.dumps(action))

    write_tree(label, os.path.join(tensor_output_folder, 'label.csv'),
               ['player', 'filename', 'chessman', 'state', 'x', 'y', 'class'])
    teams = [
        {
            "team_id": 0,
            "score": players[2]["score"] + players[4]["score"],
            "players": [players[2], players[4]]
        },
        {
            "team_id": 1,
            "score": players[1]["score"] + players[3]["score"],
            "players": [players[1], players[3]]
        }
    ]

    game_over = {
        "msg_name": "game_over",
        "msg_data": {
            "teams": teams
        },
        "time": int(time.time())
    }
    all_action.append(json.dumps(game_over))

    with open(squareness_output_file, 'w') as squareness_file:
        for i, line in enumerate(all_action):
            if i == 0:
                squareness_file.write('var msg = [' + line + '\n')
            else:
                squareness_file.write(',' + line + '\n')
        squareness_file.write('];')
    return teams, label


def one_task(result_queue, process_one_file_func, *args):
    try:
        result_queue.put(process_one_file_func(*args))
    except RuntimeError:
        traceback.print_exc()


def battle(opp_model_pool_path, model_path_2, tensor_path, squareness_path, chess_num=10000, gpu_id='0', prefix='',
           is_show_tensor=False):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    is_use_cuda = torch.cuda.is_available()
    # is_use_cuda = False

    arch = 'resnet32'
    input_channel_num = 7
    num_classes = 20 * 20 * 91

    model_path_1 = pick_opp(opp_model_pool_path, model_path_2)
    print(prefix + "model_path_1: " + model_path_1)
    model_1 = load_model(arch, input_channel_num, num_classes, is_use_cuda, model_path_1)
    model_2 = load_model(arch, input_channel_num, num_classes, is_use_cuda, model_path_2)
    result_summary = {}

    chess_i = 0
    effect_chess_num = 0

    while True:
        chess_i += 1
        if chess_i % 100 == 0:
            model_path_1 = pick_opp(opp_model_pool_path, model_path_2)
            print(prefix + "model_path_1: " + model_path_1)
            model_1 = load_model(arch, input_channel_num, num_classes, is_use_cuda, model_path_1)

        time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        if chess_i % 2 == 0:
            model_defensive = model_1
            model_offensive = model_2
            file_name = prefix + '_m1_defensive_m2_offensive_' + time_str
        else:
            model_defensive = model_2
            model_offensive = model_1
            file_name = prefix + '_m2_defensive_m1_offensive_' + time_str

        tensor_output_folder = os.path.join(tensor_path, file_name)
        squareness_output_file = os.path.join(squareness_path, file_name + 'txt')
        if not os.path.isdir(tensor_output_folder):
            os.makedirs(tensor_output_folder)
        teams, label = battle_itself(model_offensive, model_defensive, tensor_output_folder,
                                     squareness_output_file, is_use_cuda, is_show_tensor)
        if teams is None:
            print("team is none")
            continue

        win_team = (teams[0] if teams[0]["score"] >= teams[1]["score"] else teams[1])

        if chess_i % 2 == 0:
            model_1_result = teams[0]
            model_2_result = teams[1]
        else:
            model_1_result = teams[1]
            model_2_result = teams[0]

        win_players = win_team["players"]
        win_players_id = [win_players[0]["player_id"], win_players[1]["player_id"]]
        if win_players_id[0] not in label or win_players_id[1] not in label:
            print("win player not in label")
            continue
        win_label = {win_players_id[0]: label[win_players_id[0]],
                     win_players_id[1]: label[win_players_id[1]]}
        if model_2_result["score"] - model_1_result["score"] >= 30 or model_1_result["score"] > model_2_result["score"]:
            effect_chess_num += 1
            write_tree(win_label, os.path.join(tensor_output_folder, 'win_label.csv'),
                       ['player', 'filename', 'chessman', 'state', 'x', 'y', 'class'])

        model_2_win_score = model_2_result["score"] - model_1_result["score"]
        if model_2_result["score"] < model_1_result["score"]:
            summary_delta = np.array([0, 1, 0, model_2_win_score, 1])
        if model_2_result["score"] > model_1_result["score"]:
            summary_delta = np.array([1, 0, 0, model_2_win_score, 1])
        if model_2_result["score"] == model_1_result["score"]:
            summary_delta = np.array([0, 0, 1, model_2_win_score, 1])

        add_node(result_summary, [model_path_2, model_path_1], summary_delta)

        if effect_chess_num > chess_num:
            break

    return result_summary


class BlokusArgs(object):
    def __init__(self, train_data, val_data, retrain, gpu_id):
        self.arch = 'resnet32'
        self.criterion = 'CrossEntropy'
        self.train_data = train_data
        self.val_data = val_data
        self.workers = 8
        self.channel_index = '0:7'
        self.epochs = 20
        self.start_epoch = 0
        self.batch_size = 256
        self.lr = 0.001
        self.schedule = [40, 90]
        self.gamma = 0.11
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.print_freq = 10
        self.checkpoint = 'reinforcement'
        self.resume = ''
        self.retrain = retrain
        self.reset_classifier = False
        self.manualSeed = None
        self.evaluate = False
        self.gpu_id = gpu_id


def pick_opp(opp_model_pool_path, except_model=None):
    opp_list = os.listdir(opp_model_pool_path)
    opp_list = [os.path.join(opp_model_pool_path, opp_model) for opp_model in opp_list]
    opp_list = [os.path.join(opp_model, 'checkpoint.pth.tar') for opp_model in opp_list]
    if except_model and except_model in opp_list:
        opp_list.remove(except_model)
    while True:
        model_opp = random.choice(opp_list)
        if os.path.isfile(model_opp):
            break
    return model_opp


def write_summary(statistic_table, squareness_path_i):
    summary_output_file = os.path.join(squareness_path_i, 'summary.csv')
    with open(summary_output_file, 'w') as summary_output_file:
        model_2_win_num_sum = 0
        model_1_win_num_sum = 0
        draw_num_sum = 0
        model_2_win_score_average_sum = 0
        result_num_sum = 0
        for k_m2, v_m2 in statistic_table.items():
            model_path_2 = k_m2
            for k_m1, v_m1 in k_m2:
                model_path_1 = k_m1
                model_2_win_num = v_m1[0]
                model_1_win_num = v_m1[1]
                draw_num = v_m1[2]
                model_2_win_score_average = v_m1[3]
                result_num = v_m1[4]
                model_2_win_num_sum += v_m1[0]
                model_1_win_num_sum += v_m1[1]
                draw_num_sum += v_m1[2]
                model_2_win_score_average_sum += v_m1[3]
                result_num_sum += v_m1[4]

                model_2_win_score_average /= result_num
                summary_output_file.write('model 1 :' + model_path_1 + '\n')
                summary_output_file.write('model 2 :' + model_path_2 + '\n')
                summary_output_file.write('model 1 win :' + str(model_1_win_num) + '\n')
                summary_output_file.write('model 2 win :' + str(model_2_win_num) + '\n')
                summary_output_file.write('draw num :' + str(draw_num) + '\n')
                summary_output_file.write('model 2 win score average :' + str(model_2_win_score_average) + '\n')
        model_2_win_score_average_sum /= result_num_sum
        summary_output_file.write('model 1 win all:' + str(model_1_win_num_sum) + '\n')
        summary_output_file.write('model 2 win all:' + str(model_2_win_num_sum) + '\n')
        summary_output_file.write('draw num all:' + str(draw_num_sum) + '\n')
        summary_output_file.write('model 2 win score average all:' + str(model_2_win_score_average_sum) + '\n')


def reinforce_learning(argv):
    opp_model_pool_path = argv[1]
    model_path_2 = argv[2]
    tensor_path = argv[3]
    squareness_path = argv[4]
    last_val_file = argv[5]
    gpu_id = argv[6]

    gpu_list = gpu_id.split(',')
    gpu_list = [int(gpu) for gpu in gpu_list]

    last_tensor_path = None
    iter_num = 0
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    while True:
        iter_num += 1
        tensor_path_i = os.path.join(tensor_path, str(iter_num))
        squareness_path_i = os.path.join(squareness_path, str(iter_num))

        if not os.path.isdir(squareness_path_i):
            os.makedirs(squareness_path_i)
        print('%s: start battle.' % (datetime.datetime.now()))

        # statistic_table = battle(opp_model_pool_path, model_path_2, tensor_path_i, squareness_path_i, chess_num=10,
        #                          gpu_id='0', prefix='test', is_show_tensor=False)
        pool_num = 10
        pool = multiprocessing.Pool(pool_num)
        result_queue = multiprocessing.Manager().Queue()
        statistic_table = {}
        for process_i in range(pool_num):
            sub_gpu_id = str(gpu_list[process_i % len(gpu_list)])
            pool.apply_async(one_task, args=(result_queue, battle, opp_model_pool_path, model_path_2, tensor_path_i,
                                             squareness_path_i, 500, sub_gpu_id, str(process_i), False,))
        pool.close()
        pool.join()
        print('%s: end battle.' % (datetime.datetime.now()))
        print('%s: start merge result.' % (datetime.datetime.now()))
        while not result_queue.empty():
            value = result_queue.get(True)
            statistic_table = merge_tree(statistic_table, value)
        print('%s: end merge result.' % (datetime.datetime.now()))
        write_summary(statistic_table, squareness_path_i)

        label_file_name = 'win_label.csv'
        train_file = os.path.join(tensor_path_i, "train.csv")
        val_file = os.path.join(tensor_path_i, "val.csv")
        path_list = [tensor_path_i]
        summary = collect_tensor_and_label(path_list, [], label_file_name)
        with open(train_file, 'w') as cur_train_data_file, open(val_file, 'w') as cur_val_data_file:
            cur_train_data_file.write('filename,chessman, state, x, y, class\n')
            for line_i, line in enumerate(summary):
                line = ','.join(line) + '\n'
                cur_train_data_file.write(line)
                if line_i < 20000:
                    cur_val_data_file.write(line)
        blokus_args = BlokusArgs(train_file, last_val_file, model_path_2, gpu_id)
        print("train file: " + train_file)
        print("val file: " + val_file)
        print("retrain from " + model_path_2)
        checkpoint_path = train(blokus_args)

        new_model_path = os.path.join(checkpoint_path, 'checkpoint.pth.tar')
        model_path_2 = new_model_path
        last_val_file = val_file

        if last_tensor_path:
            shutil.rmtree(last_tensor_path)
        last_tensor_path = tensor_path_i


if __name__ == '__main__':
    try:
        reinforce_learning(sys.argv)
    except RuntimeError:
        traceback.print_exc()
