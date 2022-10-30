from torch import nn


class PolicyNN(nn.Module):
    """
    Takes the current board position as input and outputs a vector of probabilities for each possible move.
    """

    def __init__(self):
        super(PolicyNN, self).__init__()
        pass

    def forward(self, x):
        pass


class ValueNN(nn.Module):
    """
    Takes the current board position as input and outputs a scalar value representing the expected reward
    """

    def __init__(self):
        super(ValueNN, self).__init__()
        pass

    def forward(self, x):
        pass


class ChessNN(nn.Module):
    """
    The full model
    """
    def __init__(self):
        super(ChessNN, self).__init__()
        self.policy = PolicyNN()
        self.value = ValueNN()

    def forward(self, x):
        pass
