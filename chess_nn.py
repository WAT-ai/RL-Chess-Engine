from torch import nn

class MainNN(nn.Module):
    """
    Takes the current board position as input and outputs a residual tower
    A convolution block followed by 19 or 39 residual blocks
    """

    def __init__(self):
        super(MainNN, self).__init__()
        self.policy = PolicyNN()
        self.value = ValueNN()

    def forward(self, x):
        pass
    
class PolicyNN(nn.Module):
    """
    Takes the output from residual tower as input and outputs a vector of probabilities for each possible move.
    Consists of two layers (convolution layer and fully-connected linear layer)
    """

    def __init__(self):
        super(PolicyNN, self).__init__()
        pass

    def forward(self, x):
        pass


class ValueNN(nn.Module):
    """
    Takes the output from residual tower as input and outputs a scalar value representing the expected reward
    Consists of three layers (convolution layer and two fully-connected linear layer)
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
        self.main = MainNN()

    def forward(self, x):
        pass
