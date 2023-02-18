from env import Env, policy_predictions_to_move_probabilities
from nn import Connect4NN
import torch
from tree_search import MCTS
from tqdm import tqdm


def play_game(model, display=False):
    """
    play a game between two agents, keep track of inputs and outputs to the neural network
    :param model:
    :return:
    """
    env = Env()
    outputs = []
    mcts = MCTS(env.state, model, policy_predictions_to_move_probabilities, training_mode=True)
    while env.get_reward() is None:
        possible_moves = env.get_possible_moves()
        possible_states = [env.get_state_after_move(move)[0] for move in possible_moves]
        mcts.search(10)
        best_move = mcts.current_state.get_best_move()
        env.step(best_move)
        mcts.update_state(best_move)
        print(env.white_to_move)
        if display:
            env.display()

    return env.get_reward(), outputs


def train():
    """
    Train the model.
    :return:
    """
    memory = []
    model = Connect4NN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    for i in tqdm(range(1000)):
        optimizer.zero_grad()
        reward, outputs = play_game(model)

        for output in outputs:
            policy_label = torch.tensor([1 if i == torch.argmax(output) else 0 for i in range(7)], dtype=torch.float)
            value_label = torch.tensor([reward], dtype=torch.float)
            policy, value = output
            policy_loss = criterion(policy, policy_label)
            value_loss = criterion(value, value_label)
            loss = policy_loss + value_loss
            loss.backward()

        optimizer.step()

    return model


if __name__=="__main__":
    model = train()