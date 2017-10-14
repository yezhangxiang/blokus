import numpy as np


def matrix2tensor(chessboard_player_id, chessboard_chessman_id, chessboard_hand_no):
    channel_max_num = 10
    channel_size = 20
    tensor = np.zeros((channel_max_num, channel_size, channel_size))
    channel_i = 0

    tensor[channel_i, :, :] = chessboard_player_id
    channel_i += 1
    tensor[channel_i, :, :] = get_across_corners(chessboard_player_id, 1)
    channel_i += 1
    tensor[channel_i, :, :] = get_team(chessboard_player_id)
    channel_i += 1
    tensor[channel_i, :, :] = get_across_corners(chessboard_player_id, 1) + get_across_corners(chessboard_player_id, 3)
    channel_i += 1
    tensor[channel_i, :, :] = get_opponent(chessboard_player_id)
    channel_i += 1
    tensor[channel_i, :, :] = get_across_corners(chessboard_player_id, 2) + get_across_corners(chessboard_player_id, 4)
    channel_i += 1
    tensor[channel_i, :, :] = np.ones((channel_size, channel_size))
    channel_i += 1
    tensor[channel_i, :, :] = chessboard_chessman_id
    channel_i += 1
    tensor[channel_i, :, :] = chessboard_hand_no
    channel_i += 1
    return tensor[:channel_i, :, :]


def get_across_corners(matrix, player_id, mark_symbol=1):
    channel_size = 20
    across_corners_matrix = np.zeros((channel_size, channel_size))
    for i in range(channel_size - 1):
        for j in range(channel_size - 1):
            if matrix[i][j] == player_id and matrix[i + 1][j] != player_id and matrix[i][j + 1] != player_id and \
                            matrix[i + 1][j + 1] == 0:
                across_corners_matrix[i + 1][j + 1] = mark_symbol
            if matrix[i + 1][j] == player_id and matrix[i][j] != player_id and matrix[i + 1][j + 1] != player_id and \
                            matrix[i][j + 1] == 0:
                across_corners_matrix[i][j + 1] = mark_symbol
            if matrix[i][j + 1] == player_id and matrix[i + 1][j + 1] != player_id and matrix[i][j] != player_id and \
                            matrix[i + 1][j] == 0:
                across_corners_matrix[i + 1][j] = mark_symbol
            if matrix[i + 1][j + 1] == player_id and matrix[i + 1][j] != player_id and matrix[i][j + 1] != player_id and \
                            matrix[i][j] == 0:
                across_corners_matrix[i][j] = mark_symbol
    return across_corners_matrix


def get_opponent(matrix):
    channel_size = 20
    opponent_matrix = np.zeros((channel_size, channel_size))
    opponent_matrix[matrix == 2] = 1
    opponent_matrix[matrix == 4] = 1
    return opponent_matrix


def get_team(matrix):
    channel_size = 20
    team_matrix = np.zeros((channel_size, channel_size))
    team_matrix[matrix == 1] = 1
    team_matrix[matrix == 3] = 1
    return team_matrix
