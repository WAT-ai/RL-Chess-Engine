import torch
from chess_nn import ChessNN
from chess_env import ChessEnv, state_to_alpha_zero_input
import chess
from tree_search import MCTS
from helpers import move_probabilities_to_policy, policy_to_move_probabilities
from chess_nn import ChessNN

chess_env_2 = ChessEnv()

white = ChessNN()

mcts_white = MCTS(chess_env_2, white, state_to_alpha_zero_input, policy_to_move_probabilities)

print(mcts_white)

mcts_white.search()
