from chess_nn import ChessNN
from chess_env import ChessEnv, state_to_alpha_zero_input
from helpers import policy_to_move_probabilities
from chess_nn import ChessNN
from tree_search import MCTS
from train import play_game, train
import torch

chess_env_2 = ChessEnv()

white = ChessNN()

test_tensor = torch.randn(76, 8, 8)

move_probabilities = (policy_to_move_probabilities(test_tensor))

for move in move_probabilities.keys():
    try:
        print(move)
    except:
        pass

# train()

# model_output = white(state_to_alpha_zero_input(chess_env_2).unsqueeze(0))[1]

# result = policy_to_move_probabilities(model_output.detach().numpy())

# for move in result.keys():
#     try:
#         print(move)
#     except:
#         pass

# mcts_white = MCTS(chess_env_2, white, state_to_alpha_zero_input, policy_to_move_probabilities)

# mcts_white.search()