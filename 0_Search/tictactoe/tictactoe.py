"""
Tic Tac Toe Player
"""

import math, copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    flat_list = [cel for row in board for cel in row]
    return X if flat_list.count(O) >= flat_list.count(X) else O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possible_actions = set(
        [
            (row_index, move)
            for row_index, row in enumerate(board)
            for move, cell in enumerate(row)
            if cell == EMPTY
        ]
    )
    return possible_actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if action not in actions(board):
        raise ValueError("action not possible")

    (row, column) = action
    resulting_board = copy.deepcopy(board)
    resulting_board[row][column] = player(board)
    return resulting_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    rows = board
    cols = [list(col) for col in zip(*board)]
    diag1 = [board[i][i] for i in range(3)]
    diag2 = [board[i][2 - i] for i in range(3)]
    lines = rows + cols + [diag1] + [diag2]
    for line in lines:
        player = are_all_cells_same_player(line)
        if player is not None:
            return player
    return None


def are_all_cells_same_player(cels):
    winner_player = set(cels)
    if len(winner_player) == 1 and None not in winner_player:
        return winner_player.pop()
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    return True if winner(board) or not actions(board) else False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    if winner(board) == O:
        return -1
    return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None
    current_player = player(board)
    chosen_action = (0, 0)
    alpha = float("-inf")
    beta = float("inf")
    if current_player == X:
        value = float("-inf")
    else:
        value = float("inf")

    for action in actions(board):
        new_board = result(board, action)
        if current_player == X:
            new_value = minValue(new_board, alpha, beta)
            alpha = max(alpha, new_value)
            if new_value > value:
                value = new_value
                chosen_action = action
        if current_player == O:
            new_value = maxValue(new_board, alpha, beta)
            beta = min(beta, new_value)
            if new_value < value:
                value = new_value
                chosen_action = action

    return chosen_action


def maxValue(board, alpha, beta):
    if terminal(board):
        return utility(board)
    value = float("-inf")
    for action in actions(board):
        value = max(value, minValue(result(board, action), alpha, beta))
        alpha = max(alpha, value)
        if value >= beta:
            return value
    return value


def minValue(board, alpha, beta):
    if terminal(board):
        return utility(board)
    value = float("inf")
    for action in actions(board):
        value = min(value, maxValue(result(board, action), alpha, beta))
        beta = min(beta, value)
        if value <= alpha:
            return value
    return value
