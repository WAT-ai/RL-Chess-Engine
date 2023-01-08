import torch
from torch import nn


class Connect4NN(nn.Module):
    """
    The full model
    """

    def __init__(self):
        super(Connect4NN, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.l1 = nn.Linear(64 * 7 * 6, 128)
        self.l2 = nn.Linear(128, 8)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(-1, 64 * 7 * 6)
        x = nn.functional.relu(self.l1(x))
        x = self.l2(x)
        policy = nn.functional.softmax(x[:, :7], dim=1)
        value = torch.tanh(x[:, 7])
        return policy, value


if __name__ == '__main__':
    model = Connect4NN()
    print(model)
    print(model(torch.rand(1, 2, 6, 7)))
