from torch import nn


class PolicyNN(nn.Module):
    """
    Takes the current board position as input and outputs a vector of probabilities for each possible move.
    """

    def __init__(self):
        super(PolicyNN, self).__init__()
        self.network = nn.Sequential(
            '''
            (1) A convolution of 2 filters of kernel size 1x1 with stride 1
                - I am not sure the difference between convolutional filter and layer
                - Unclear on inputs/outputs; typically input = batch size
            (2) Batch normalization
            (3) A rectifier nonlinearity
            (4) A fully connected linear layer that outputs a vector of size 19^2 + 1 = 362, corresponding to logit 
                probabilities for all intersections and the pass move
                - Unclear on inputs/outputs
                - I think the second half of this is about the game of GO
            '''

            # Layer 1
            nn.Conv2d(in_channels = , out_channels = , kernel_size = 1, stride = 1), 
            nn.BatchNorm2d(num_features = out_channels),
            nn.ReLU(),

            # Layer 2
            nn.Conv2d(in_channels = batch size, out_channels = , kernel_size = 1, stride = 1),
            nn.BatchNorm2d(num_features = out_channels),
            nn.ReLU()

            # Fully Connected Layer 
            nn.Linear(in_features = , out_features = 362) 
        )

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
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            """
            Residual Block
            Make 39 of these residual blocks without repeating
            """
            nn.Conv2d(256, 256, 32, kernel_size = 3, stride = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 32, kernel_size = 3, stride = 1),
            nn.BatchNorm2d(256),
            ''' Implement skip connection layer'''
            nn.ReLU()
            
            
        )
        self.policy = PolicyNN()
        self.value = ValueNN()

    def forward(self, x):
        pass
