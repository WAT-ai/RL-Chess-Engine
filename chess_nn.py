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
        self.network = nn.Sequential(
            """
            First Convolutional Block
            """
            nn.Conv2d(ask thomas, 256, 32, kernel_size = 3, stride = 1),
            nn.BatchNorm2d(256)
            nn.ReLU(),
            
            """
            Residual Block
            Make 39 of these residual blocks without repeating
            """
            nn.Conv2d(256, 256, 32, kernel_size = 3, stride = 1),
            nn.BatchNorm2d(256)
            nn.ReLU(),
            nn.Conv2d(256, 256, 32, kernel_size = 3, stride = 1),
            nn.BatchNorm2d(256),
            ''' Implement skip connection layer'''
            nn.ReLU(),
            
            
        )
        self.policy = PolicyNN()
        self.value = ValueNN()

    def forward(self, x):
        pass
