from env import Env
from nn import Connect4NN
import torch
from tree_search import get_move_scores
from tqdm import tqdm


def play_game(model, display=False):
    """
    play a game between two agents, keep track of inputs and outputs to the neural network
    :param model:
    :return:
    """
    env = Env()
    outputs = []
    while env.get_reward() is None:
        possible_moves = env.get_possible_moves()
        possible_states = [env.get_state_after_move(move)[0] for move in possible_moves]
        scores = get_move_scores(model, possible_states)
        outputs.append(scores)
        best_move = possible_moves[torch.argmax(scores) if env.white_to_move else torch.argmin(scores)]
        env.step(best_move)
        print(env.white_to_move)
        if display:
            env.display()

    return env.get_reward(), outputs


def train():
    """
    Train the model.
    :return:
    """
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