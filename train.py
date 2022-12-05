from chess_env import ChessEnv
from chess_nn import ChessNN
from torch import optim
from tree_search import MCTS

NUM_SIMULATIONS = 100

def play_game(white, black=None):
    """
    play a game between two agents, keep track of inputs and outputs to the neural network
    :param model:
    :return:
    """
    chess_env = ChessEnv() 
    mcts_white = MCTS()
    mcts_black = MCTS() if black is not None else None
    # get_reward() returns 1 if win for white, 0 for draw, -1 for loss for white, None if game is not over
    while(chess_env.get_reward() is None):
        current_mcts = mcts_white
        current_nn = white
        # chess_env.turn is true if white to move, false if black to move
        if not chess_env.turn and mcts_black is not None:
            current_nn = black
            current_mcts = mcts_black

        current_mcts.search(NUM_SIMULATIONS, current_nn)
        chess_env.step(current_mcts.get_best_move())
        mcts_white.update_root_node(chess_env.get_state())
        if mcts_black is not None:
            mcts_black.update_root_node(chess_env.get_state())

    return chess_env.get_reward()

def train():
    """
    train the model through self-play
    :return:
    """
    pass



