import torch
from torch import nn



class PolicyNN(nn.Module):
    """
    Takes the current board position as input and outputs a vector of probabilities for each possible move.
    """
    def __init__(self):
        super(PolicyNN, self).__init__()
        

        self.convolution = nn.Conv2d(256, 73, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(73)
        self.soft_max = nn.Softmax()


    def forward(self, x):
        output = self.convolution(x)
        output = self.bn(output)
        output = self.soft_max(output)

        return output


class ValueNN(nn.Module):
    """
    Takes the current board position as input and outputs a scalar value representing the expected reward
    """
    def __init__(self):
        super(ValueNN, self).__init__()
        
        self.convolution = nn.Conv2d(256, 1, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()

        self.fcl = nn.Linear(8 * 8, 32)
        self.fcl2 = nn.Linear(32, 1)

        self.tanh = nn.Tanh()

        
    def forward(self, x):
        output = self.convolution(x)
        output = self.bn(output)
        output = self.relu(output)

        output = torch.flatten(output,start_dim=1)
        output = self.fcl(output)
        output = self.relu(output)
        output = self.fcl2(output)
        output = self.tanh(output)

        return output
        


class ChessNN(nn.Module):
    """
    The full model
    """
    def __init__(self):
        super(ChessNN, self).__init__()

        self.convBlock = ConvBlock()   
        self.res_blocks = [ResBlock() for i in range(19)]
        self.policy = PolicyNN()
        self.value = ValueNN()

    def forward(self, x):
        output = self.convBlock(x)
        for block in self.res_blocks:
            output = block(output)
        
        output_copy = output.clone()
        policy_output = self.policy(output_copy)
        value_output = self.value(output)

        return value_output, policy_output

class ConvBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.convNet = nn.Conv2d(19, 256, kernel_size = 3,  stride=1, padding="same")
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.convNet(x)
        output = self.bn(output)
        output = self.relu(output)

        return output
        
       

class ResBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.convNet1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding="same")
        self.bn1_2D = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU()

        
        self.convNet2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding="same")
        self.bn2_2D = nn.BatchNorm2d(256)
        

    def forward(self, x):

        residual = x.clone();

        output = self.convNet1(x)
        output = self.bn1_2D(output)
        output = self.relu1(output)
        output = self.convNet2(output)
        output = self.bn2_2D(output)
        output += residual
        output = self.relu1(output)

        return output



if(__name__ == "__main__"):
    input = torch.rand((1, 12, 8, 8))
    chessNN = ChessNN()

    output = chessNN(input)

    print(output[0].shape, output[1].shape)




