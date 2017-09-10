import json
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import sys

from utility import process_file_list, show_channels
from multiway_tree import add_node, write_tree
from matrix2tensor import matrix2tensor

channel_size = 20


def json2tensor(json_file, chessman_dic, chessman_state_id, output_folder, is_show_tensor=False):
    with open(json_file) as json_f:
        file_name = os.path.basename(json_file)
        sub_output_folder = os.path.join(output_folder, file_name)
        if not os.path.exists(sub_output_folder):
            os.makedirs(sub_output_folder)
        label = {}
        chessboard_player_id = np.zeros((channel_size, channel_size))
        chessboard_chessman_id = np.zeros((channel_size, channel_size))
        chessboard_hand_no = np.zeros((channel_size, channel_size))
        for line in json_f:
            start_index = line.find('{')
            if start_index == -1:
                continue
            line_sub = line[start_index:]
            msg = json.loads(line_sub)
            if msg['msg_name'] == 'game_start':
                pass
            elif msg['msg_name'] == 'notification':
                msg_data = msg['msg_data']
                player_id = msg_data['player_id']
                chessman = msg_data['chessman']
                if 'id' not in chessman:
                    continue
                chessman_id = chessman['id']
                squareness = chessman['squareness']
                hand_no = msg_data['hand']['no']

                chessboard_player_id_regulated = regular_chessboard(chessboard_player_id, player_id,
                                                                    is_replace_player_id=True)
                chessboard_chessman_id_regulated = regular_chessboard(chessboard_chessman_id, player_id)
                chessboard_hand_no_regulated = regular_chessboard(chessboard_hand_no, player_id)
                for grid in squareness:
                    chessboard_player_id[grid['x']][grid['y']] = player_id
                    chessboard_chessman_id[grid['x']][grid['y']] = chessman_id
                    chessboard_hand_no[grid['x']][grid['y']] = hand_no
                chessboard_player_id_regulated_new = regular_chessboard(chessboard_player_id, player_id,
                                                                          is_replace_player_id=True)
                chessman_in_board_regulated = chessboard_player_id_regulated_new - chessboard_player_id_regulated
                squareness_regulated = chessman_in_board2squareness(chessman_in_board_regulated)
                regular_chessman, regular_x, regular_y = squareness2regular_chessman(squareness_regulated)
                state = find_chessman(chessman_dic[chessman_id], regular_chessman)

                tensor = matrix2tensor(chessboard_player_id_regulated,
                                       chessboard_chessman_id_regulated,
                                       chessboard_hand_no_regulated)
                show_channels(tensor, is_show_tensor, True)
                tensor_save_file = os.path.join(sub_output_folder, str(hand_no) + 'npy')
                np.save(tensor_save_file, tensor)
                add_node(label, [player_id, tensor_save_file],
                         [chessman_id, state, regular_x, regular_y,
                          channel_size * regular_y + regular_x +
                          channel_size * channel_size * chessman_state_id[chessman_id][state]])
            elif msg['msg_name'] == 'game_over':
                if 'msg_data' not in msg:
                    continue
                msg_data = msg['msg_data']
                teams = msg_data['teams']
                win_team = (teams[0] if teams[0]['score'] >= teams[1]['score'] else teams[1])
                win_players = win_team['players']
                win_players_id = [win_players[0]['player_id'], win_players[1]['player_id']]
                if win_players_id[0] not in label or win_players_id[1] not in label:
                    continue
                win_lable = {win_players_id[0]: label[win_players_id[0]],
                             win_players_id[1]: label[win_players_id[1]]}
                write_tree(win_lable, os.path.join(sub_output_folder, 'win_label.csv'),
                           ['player', 'filename', 'chessman', 'state', 'x', 'y', 'class'])

        write_tree(label, os.path.join(sub_output_folder, 'label.csv'),
                   ['player', 'filename', 'chessman', 'state', 'x', 'y', 'class'])


def regular_chessboard(chessboard, player_id, is_replace_player_id=False):
    rot_k = player_id - 1
    chessboard_regulated = chessboard.copy()
    if is_replace_player_id:
        chessboard_regulated = np.zeros((channel_size, channel_size))
        for i in range(1, 5):
            regular_id = (i - rot_k) % 4
            if regular_id == 0:
                regular_id = 4
            chessboard_regulated[chessboard == i] = regular_id
    chessboard_regulated = np.rot90(chessboard_regulated, rot_k)
    return chessboard_regulated


def chessman_in_board2squareness(chessman_in_board):
    squareness = []
    for i in range(channel_size):
        for j in range(channel_size):
            if chessman_in_board[i][j] != 0:
                squareness.append({'x': i, 'y': j})
    return squareness


def squareness2regular_chessman(squareness):
    x = [grid['x'] for grid in squareness]
    y = [grid['y'] for grid in squareness]
    x_min = min(x)
    y_min = min(y)
    x_regular = [x_i - x_min for x_i in x]
    y_regular = [y_i - y_min for y_i in y]
    chesssman = np.zeros((max(x_regular) + 1, max(y_regular) + 1))
    for i in range(len(x_regular)):
        chesssman[x_regular[i]][y_regular[i]] = 1
    return chesssman, x_min, y_min


def extend_all_chessman(is_show=False):
    chessman_dic = {
        101: [np.array([[1]])],
        201: [np.array([[1, 1]])],
        301: [np.array([[1, 1, 1]])],
        302: [np.array([[1, 1], [1, 0]])],
        401: [np.array([[1, 1, 1, 1]])],
        402: [np.array([[0, 1, 0], [1, 1, 1]])],
        403: [np.array([[1, 1, 0], [0, 1, 1]])],
        404: [np.array([[1, 0, 0], [1, 1, 1]])],
        405: [np.array([[1, 1], [1, 1]])],
        501: [np.array([[1, 1, 1, 1, 1]])],
        502: [np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]])],
        503: [np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])],
        504: [np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]])],
        505: [np.array([[0, 0, 1], [0, 1, 1], [1, 1, 0]])],
        506: [np.array([[1, 1, 1], [0, 1, 1]])],
        507: [np.array([[1, 1, 1, 1], [0, 1, 0, 0]])],
        508: [np.array([[1, 0, 0, 0], [1, 1, 1, 1]])],
        509: [np.array([[1, 1, 1], [1, 0, 1]])],
        510: [np.array([[0, 0, 1, 1], [1, 1, 1, 0]])],
        511: [np.array([[0, 1, 1], [0, 1, 0], [1, 1, 0]])],
        512: [np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]])],
    }

    for name, chessman_list in chessman_dic.items():
        chessman = chessman_list[0]
        for i in range(8):
            chessman = np.rot90(chessman, i)
            if i == 4:
                chessman = np.transpose(chessman)
            if find_chessman(chessman_list, chessman) == -1:
                chessman_list.append(chessman)
    if is_show:
        for name, chessman_list in chessman_dic.items():
            channel_num = len(chessman_list)
            row = int(math.sqrt(channel_num))
            column = int(math.ceil(channel_num / row))
            fig, axes = plt.subplots(row, column)
            fig.canvas.set_window_title(str(name))
            for i, chessman in enumerate(chessman_list):
                cur_ax = axes if channel_num == 1 else axes[i] if channel_num < 3 else axes[
                    int(i / column), int(i % column)]
                cur_ax.imshow(chessman)
            plt.show()
    return chessman_dic


def get_chessman_state_index(chessman_dic):
    chessman_state_id = {}
    chessman_state_id_inverse = {}
    dic_sorted = sorted(chessman_dic.items(), key=lambda d: d[0])
    index = 0
    for item in dic_sorted:
        name = item[0]
        chessman_list = item[1]
        for state, chessman_item in enumerate(chessman_list):
            add_node(chessman_state_id, [name, state], index)
            add_node(chessman_state_id_inverse, [index], [name, state])
            index += 1
    return chessman_state_id, chessman_state_id_inverse


def find_chessman(chessman_list, chessman):
    for i, chessman_item in enumerate(chessman_list):
        if chessman.shape == chessman_item.shape and np.sum(np.abs(chessman - chessman_item)) == 0:
            return i
    return -1


if __name__ == '__main__':
    chessman_dic = extend_all_chessman()
    chessman_state_id, _ = get_chessman_state_index(chessman_dic)
    for argv_i in range(1, len(sys.argv), 2):
        process_file_list(sys.argv[argv_i], json2tensor, True, chessman_dic, chessman_state_id, sys.argv[argv_i + 1],
                        False)
