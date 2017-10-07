import traceback
import socket
import numpy as np
import json
import torch
import os
import time
import sys
import models

from generate_tensor import chessman_in_board2squareness, regular_chessboard, extend_all_chessman, \
    get_chessman_state_index
from matrix2tensor import matrix2tensor
import random


def get_squareness(chessman_dic, channel_size, chessman_id, state, x, y, player_id):
    chessman = chessman_dic[chessman_id][state]
    board = np.zeros((channel_size, channel_size))
    for i in range(chessman.shape[0]):
        for j in range(chessman.shape[1]):
            if chessman[i][j] != 0:
                board[x + i][y + j] = 1
    rot_k = 5 - player_id
    if rot_k == 4:
        rot_k = 0
    board = np.rot90(board, rot_k)
    squareness = chessman_in_board2squareness(board)
    return squareness


def play(chessman_dic, chessman_state_id_inverse, tensor, model, player_id, is_use_cuda,
         maxk=5, random_choose_rate=0.2):
    # switch to evaluate mode
    model.eval()

    channel_size = 20
    channel_num = 7
    input_tensor = torch.from_numpy(tensor[0:channel_num, :, :])
    input_tensor = input_tensor.view(1, channel_num, 20, 20)
    input_all = torch.from_numpy(tensor)

    input_tensor = input_tensor.float()
    if is_use_cuda:
        input_tensor = input_tensor.cuda()
    input_tensor = torch.autograd.Variable(input_tensor, volatile=True)

    # compute output
    output = model(input_tensor)

    chessman_id, state, x, y = get_legal_solution(input_all, output.data,
                                                  chessman_dic, chessman_state_id_inverse,
                                                  maxk=maxk, player_id=1, random_choose_rate=random_choose_rate)
    chessman = {}
    if chessman_id:
        squareness = get_squareness(chessman_id, state, x, y, player_id)
        chessman = {
            "id": chessman_id,
            "squareness": squareness
        }
    return chessman, chessman_id, state, x, y


def json2msg(js):
    js = json.dumps(js)
    js_len = str(len(js))
    js = '0' * (5 - len(js_len)) + js_len + js
    return js.encode()


def load_model(arch, input_channel_num, num_classes, is_use_cuda, resume_path):
    model = models.__dict__[arch](
        input_channel_num=input_channel_num,
        num_classes=num_classes, )
    if is_use_cuda:
        model = torch.nn.DataParallel(model).cuda()

    assert os.path.isfile(resume_path), 'Error: no checkpoint directory found!'
    if is_use_cuda:
        checkpoint = torch.load(resume_path)
    else:
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
        state_dict = checkpoint['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        checkpoint['state_dict'] = new_state_dict
    model.load_state_dict(checkpoint['state_dict'])
    return model


def battle_via_server(argv):
    my_team_id = int(argv[1])
    host = argv[2]
    port = int(argv[3])

    gpu_id = argv[4]
    resume_path = argv[5]

    arch = 'resnet32'
    input_channel_num = 7
    num_classes = 20 * 20 * 91
    channel_size = 20

    chessman_dic = extend_all_chessman()
    _, chessman_state_id_inverse = get_chessman_state_index(chessman_dic)

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    is_use_cuda = torch.cuda.is_available()
    # is_use_cuda = False

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))

    chessboard_player_id = np.zeros((channel_size, channel_size))
    chessboard_chessman_id = np.zeros((channel_size, channel_size))
    chessboard_hand_no = np.zeros((channel_size, channel_size))

    model = load_model(arch, input_channel_num, num_classes, is_use_cuda, resume_path)
    tensor = np.zeros((9, channel_size, channel_size))
    play(chessman_dic, chessman_state_id_inverse, tensor, model, 1, is_use_cuda)

    reg_msg = {
        "msg_name": "registration",
        "msg_data": {
            "team_id": my_team_id,
            "team_name": "yangxixi"
        }
    }
    s.send(json2msg(reg_msg))

    while True:
        data = s.recv(5000)
        data = data.decode()
        while len(data) > 0:
            try:
                msg_len = int(data[0:5])
                msg_string = data[5:msg_len + 5]
                data = data[(5 + msg_len):]

                msg = json.loads(msg_string)
            except RuntimeError:
                action = {
                    "msg_name": "action",
                    "msg_data": {
                        "hand_no": 1,
                        "team_id": my_team_id,
                        "player_id": player_id,
                        "chessman": {}
                    }
                }
                s.send(json2msg(action))
                print(data)
                print("load msg wrong!!!!")
            else:
                if msg["msg_name"] == "game_start":
                    my_players = {}
                    msg_data = msg["msg_data"]
                    players = msg_data["players"]
                    for i, player in enumerate(players):
                        if player["team_id"] == my_team_id:
                            my_players["player_id"] = player["birthplace"]
                elif msg["msg_name"] == "inquire":
                    start = time.time()
                    msg_data = msg["msg_data"]
                    player_id = msg_data["player_id"]
                    hand_no = msg_data["hand_no"]
                    chessboard_player_id_regulated = regular_chessboard(chessboard_player_id, player_id,
                                                                        is_replace_player_id=True)
                    chessboard_chessman_id_regulated = regular_chessboard(chessboard_chessman_id, player_id)
                    chessboard_hand_no_regulated = regular_chessboard(chessboard_hand_no, player_id)
                    tensor = matrix2tensor(chessboard_player_id_regulated,
                                           chessboard_chessman_id_regulated,
                                           chessboard_hand_no_regulated)
                    chessman, _, _, _, _ = play(chessman_dic, chessman_state_id_inverse, tensor, model, player_id,
                                                is_use_cuda)
                    action = {
                        "msg_name": "action",
                        "msg_data": {
                            "hand_no": hand_no,
                            "team_id": my_team_id,
                            "player_id": player_id,
                            "chessman": chessman
                        }
                    }
                    s.send(json2msg(action))
                    end = time.time()
                    print(json2msg(action))
                    print("hand no %d take %.2f ms" % (hand_no, (end - start) * 1000))
                elif msg["msg_name"] == "notification":
                    msg_data = msg["msg_data"]
                    player_id = msg_data["player_id"]
                    chessman = msg_data["chessman"]
                    if 'id' not in chessman:
                        continue
                    chessman_id = chessman["id"]
                    squareness = chessman["squareness"]
                    hand_no = msg_data["hand_no"]

                    for grid in squareness:
                        chessboard_player_id[grid['x']][grid['y']] = player_id
                        chessboard_chessman_id[grid['x']][grid['y']] = chessman_id
                        chessboard_hand_no[grid['x']][grid['y']] = hand_no
                elif msg["msg_name"] == "game_over":
                    s.close()
                    return
                else:
                    break

    s.close()


def legal_check(chessboard_play_id, chessboard_chessman_id, chessman_id, state, x, y, chessman_dic, player_id=1):
    chessman = chessman_dic[chessman_id][state]
    if x < 0 or y < 0 or x + chessman.shape[0] - 1 >= 20 or y + chessman.shape[1] - 1 >= 20:
        return False
    foo = chessboard_play_id == player_id
    if np.sum(chessboard_chessman_id[foo]) == 0 and x == 0 and y == 0:
        return True

    bar = chessboard_chessman_id == chessman_id
    if np.logical_and(foo, bar).any():
        return False

    for i in range(chessman.shape[0]):
        for j in range(chessman.shape[1]):
            if chessman[i][j] != 0:
                if chessboard_play_id[x + i][y + j] != 0:
                    return False
                if chessboard_play_id[x + i + 1][y + j] == player_id or \
                                chessboard_play_id[x + i - 1][y + j] == player_id or \
                                chessboard_play_id[x + i][y + j + 1] == player_id or \
                                chessboard_play_id[x + i][y + j - 1] == player_id:
                    return False

    for i in range(chessman.shape[0]):
        for j in range(chessman.shape[1]):
            if chessman[i][j] != 0:
                if chessboard_play_id[x + i + 1][y + j + 1] == player_id or \
                                chessboard_play_id[x + i - 1][y + j + 1] == player_id or \
                                chessboard_play_id[x + i + 1][y + j - 1] == player_id or \
                                chessboard_play_id[x + i - 1][y + j - 1] == player_id:
                    return True
    return False


def class2chessman(class_id, index2chessman, channel_size=20):
    chessman_state_id = int(class_id / (channel_size * channel_size))
    chessman_state_mod = class_id % (channel_size * channel_size)
    chessman_id, state = index2chessman[chessman_state_id]
    y = int(chessman_state_mod / channel_size)
    x = chessman_state_mod % channel_size
    return chessman_id, state, x, y


def legal_accuracy(inputs, ouputs, maxk=1, player_id=1):
    chessman_dic = extend_all_chessman()
    _, chessman_state_id_inverse = get_chessman_state_index(chessman_dic)
    _, pred_class_id_batch = ouputs.topk(maxk, 1, True, True)

    legal_num = [0] * maxk
    all_num = 0
    topk_exit_legal_num = 0
    hand_no_legal_num = [0] * 21
    hand_no_all_num = [0] * 21

    for i in range(len(inputs)):
        input = inputs[i]
        pred_class_id = pred_class_id_batch[i, :]
        chessboard_player_id = input[0, :, :].numpy()
        chessboard_chessman_id = input[7, :, :].numpy()
        chessboard_hand_no = input[8, :, :].numpy()

        all_hand_no = chessboard_hand_no[chessboard_player_id == player_id]
        hand_no = len(np.unique(all_hand_no))

        is_topk_exit_legal = False
        for k, class_id in enumerate(pred_class_id):
            chessman_id, state, x, y = class2chessman(class_id, chessman_state_id_inverse)
            if legal_check(chessboard_player_id, chessboard_chessman_id, chessman_state_id_inverse, state, x, y,
                           chessman_dic, player_id):
                legal_num[k] += 1
                hand_no_legal_num[hand_no] += 1
                is_topk_exit_legal = True

        all_num += 1
        hand_no_legal_num[hand_no] += 1
        if is_topk_exit_legal:
            topk_exit_legal_num += 1

        legal_acc = [legal_num_i / all_num for legal_num_i in legal_num]
        hand_no_legal_acc = [hand_no_legal_num[i] / hand_no_all_num[i] if hand_no_all_num[i] > 0 else 0 for i in
                             range(21)]
        print('{0} {1}/{2}, valid accuracy'.format(legal_num, topk_exit_legal_num, all_num) + str(legal_acc))
        return legal_acc


def get_legal_solution(input, output, chessman_dic, chessman_state_id_inverse, maxk=10, player_id=1,
                       random_choose_rate=0.2):
    _, pred_class_id_batch = output.topk(maxk, 1, True, True)

    pred_class_id = pred_class_id_batch[0, :]
    chessboard_player_id = input[0, :, :].numpy()
    chessboard_across_corners = input[1, :, :].numpy()
    chessboard_chessman_id = input[7, :, :].numpy()

    is_random_choose = False
    if random.random() < random_choose_rate:
        is_random_choose = True

    legal_solution = []
    for k, class_id in enumerate(pred_class_id):
        chessman_id, state, x, y = class2chessman(class_id, chessman_state_id_inverse)
        if legal_check(chessboard_player_id, chessboard_chessman_id, chessman_state_id_inverse, state, x, y,
                       chessman_dic, player_id):
            if not is_random_choose:
                return chessman_id, state, x, y
            else:
                legal_solution.append((chessman_id, state, x, y))
    if len(legal_solution) == 0:
        print('Top n has no legal solution')
        return search_legal_solution(chessboard_player_id, chessboard_chessman_id,
                                     chessboard_across_corners, chessman_dic, player_id)
    else:
        return random.choice(legal_solution)


def search_legal_solution(chessboard_player_id, chessboard_chessman_id,
                          chessboard_across_corners, chessman_dic, player_id):
    used_chessman_id = chessboard_chessman_id[chessboard_player_id != 0]
    used_chessman_id = np.unique(used_chessman_id)
    unused_chessman_id = []
    for key in chessman_dic:
        if key not in used_chessman_id:
            unused_chessman_id.append(key)
    for i in range(chessboard_across_corners.shape[0]):
        for j in range(chessboard_across_corners.shape[1]):
            if chessboard_across_corners[i][j] == 0:
                continue
            for chessman_id in unused_chessman_id:
                for state, chessman in enumerate(chessman_dic[chessman_id]):
                    for chessman_i in range(chessman.shape[0]):
                        for chessman_j in range(chessman.shape[1]):
                            if chessman[chessman_i][chessman_j] == 0:
                                continue
                            x = i - chessman_i
                            y = j - chessman_j
                            if legal_check(chessboard_player_id, chessboard_chessman_id, chessman_id, state, x, y,
                                           chessman_dic, player_id):
                                return chessman_id, state, x, y
    print('There is no legal solution')
    return None, None, None, None


if __name__ == '__main__':
    try:
        battle_via_server(sys.argv)
    except RuntimeError:
        traceback.print_exc()
