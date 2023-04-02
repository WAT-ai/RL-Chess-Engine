import torch
import torch.nn.functional as F
from chess_env import ChessEnv, state_to_alpha_zero_input
from chess_nn import ChessNN
from torch import optim
from tree_search import MCTS
from tree_search import Node
from helpers import move_probabilities_to_policy, policy_to_move_probabilities

NUM_EPISODES = 1000
LEARNING_RATE = 0.001

def play_game(white, black=None):
    """
    play a game between two agents, keep track of inputs and outputs to the neural network
    :param model:
    :return:
    """
    chess_env = ChessEnv() 
    mcts_white = MCTS(ChessEnv(), white, state_to_alpha_zero_input, policy_to_move_probabilities)
    mcts_black = MCTS(ChessEnv(), black, state_to_alpha_zero_input, policy_to_move_probabilities) if black is not None else None
    # get_reward() returns 1 if win for white, 0 for draw, -1 for loss for white, None if game is not over
    while(chess_env.get_reward() is None):
        current_mcts = mcts_white
        current_nn = white
        # chess_env.turn is true if white to move, false if black to move
        if not chess_env.turn and mcts_black is not None:
            current_nn = black
            current_mcts = mcts_black

        current_mcts.search()
        move = current_mcts.current_node.select_move()
        # print(f"\n {mcts_white.current_node.board_state}")
        # print(move)
        chess_env.step(move)
        mcts_white.update_state(move)
        if mcts_black is not None:
            mcts_black.update_state(chess_env.get_state())

    return_values = (chess_env.get_reward(), mcts_white, mcts_black) if black is not None else (chess_env.get_reward(), mcts_white)
    return return_values

def train():
    """
    train the model through self-play
    :return:
    """
    chess_nn = ChessNN()
    optimizer = optim.Adam(chess_nn.parameters(), lr=LEARNING_RATE)
    experiences = []
    for _ in range(NUM_EPISODES): 
        episode_reward, mcts_white = play_game(chess_nn)
        print(episode_reward)
        store_experiences(mcts_white.current_node, episode_reward, experiences)
        # Passes a list of size (num_experiences, 19, 8, 8) to chess_nn
        predicted_value, predicted_policy = chess_nn(torch.stack([state_to_alpha_zero_input(node.board_state) for node in experiences]))
        target_value = torch.tensor([node.reward for node in experiences]).unsqueeze(1).float()
        target_policy = torch.tensor([move_probabilities_to_policy(node.get_move_probabilities()) for node in experiences]).float()
        loss_value = F.mse_loss(predicted_value, target_value)
        loss_policy = F.mse_loss(predicted_policy, target_policy)
        loss = loss_value + loss_policy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.save(chess_nn.state_dict(), "model.pt")

def store_experiences(node: Node, episode_reward: int, experiences: list):
    # Parent of root node will be none
    if node is not None:
        node.reward = abs(episode_reward)
        experiences.append(node)
        # Next layer of MCTS will correspond to opposite color, i.e. reward of parent node should be the opposite of child node
        store_experiences(node.parent, node.reward * -1, experiences)



