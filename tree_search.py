
import math
import random
from typing import Optional
from chess_env import *

class MCTS:
    def __init__(self, *args, **kwargs):

        pass

    def search(self, *args, **kwargs):
        pass

class Node:
    def __init__(self) -> None:
        self.environment = ChessEnv()
        self.w_i = 0
        self.n_i = 0
        self.n = 0
        self.parent = None
        self.children = set()

# https://www.chessprogramming.org/UCT
# https://en.wikipedia.org/wiki/Monte_Carlo_tree_search#Exploration_and_exploitation
# w_i : the number of wins for the node after the i-th move
# n : the number of times the parent has been visited (total number of simulations after i_th move run by parent node)
# n_i: the number of simulations for the node after the i_th move
# c : exploration parameter, theoretically sqrt(2) 
# The first component corresponds to exploitation, as it is high for moves with high average win ratio
# The second component corresponds to exploration, as it is high for few simulations 
# Thus, in selection, we maximize this value
def ucb_score(node : Node, c = math.sqrt(2)):
    return node.w_i / node.n + c * math.sqrt(math.log(node.n, math.e) / node.n_i) 

# Select child node i such that ucb1 is maximized
def selection(node : Node) -> Node:
    selected_child = node
    max_ucb = -math.inf
    for child in node.children:
        child_ucb = ucb_score(child)
        if (child_ucb > max_ucb):
            max_ucb = child_ucb
            selected_child = child
    return selected_child

# Continue creating child nodes, and picking the child with highest UCB
def expansion(node : Node) -> Node:
    # Return current node if leaf node
    if (len(node.children) == 0):
        return node
    selected_child = node
    max_ucb = -math.inf
    for child in node.children:
        child_ucb = ucb_score(child)
        if (child_ucb > max_ucb):
            max_ucb = child_ucb
            selected_child = child
    return expansion(selected_child)
    
# A simulation is performed by choosing uniformly random moves until the game ends
# by draw, win, or loss
def rollout(node : Node) -> tuple[Optional[float], Node]:
    if (node.environment.get_reward() != None):
        return (node.environment.get_reward(), node)
    
    # get list of all possible moves from current state
    # use the policy network (given a state, output a probability distribution over all possible moves)
    child = random.choice(node.environment.get_possible_moves())
    return rollout(child)

# Use result of rollout and update parent node
def backpropagation():
    pass
