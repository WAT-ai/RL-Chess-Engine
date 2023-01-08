import torch


class MCTS:
    """
    Monte Carlo Tree Search.
    Creates a game tree and searches it for the best continuation.

    Example usage:
    mcts = MCTS(initial_state)
    while game is not over:
        mcts.search(1000, model)
        best_move = mcts.get_best_move()
        mcts.update_root_node(best_move)

    """
    def __init__(self, root_node):
        pass

    def search(self, simulations, model):
        """
        Search the game tree for the best continuation.
        :param simulations: number of moves from root to check
        :param model: model to evaluate the state, should output policy and value
        :return: None
        """
        pass

    def get_best_move(self, move):
        """
        Get the best move from the current root node based on the search.
        :param move:
        :return:
        """
        pass

    def update_root_node(self, state: str):
        """
        Update the root node to the given state. Prune the tree of nodes not descended from the new root.
        :param state:
        :return:
        """
        pass


def get_move_scores(model, possible_states):
    """
    Get the scores (values) for each possible move.
    :param model: model to evaluate the state, should output policy and value
    :param possible_states: states to check
    :return:
    """
    values = []
    for state in possible_states:
        value = model(torch.tensor(state, dtype=torch.float))[1]
        values.append(value)
    return torch.tensor(values)
