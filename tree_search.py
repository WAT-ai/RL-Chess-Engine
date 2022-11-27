
import math
from chess_env import *

num_simulations = 100
stop_threshold = 0.1

class MCTS:
    def __init__(self, board_state, value_model, policy_model):
        self.board_state = board_state
        self.value_model = value_model
        self.policy_model = policy_model

    def search(self):
        """
        MCTS algorithm, select, expand, backpropagate.
        """
        root = Node(self.board_state, None, 0)

        move_probabilities = self.policy_model.predict(self.board_state) 
        root.expand(move_probabilities)

        for _ in range(num_simulations):
            node = root
            
            # Select a leaf node
            while node.is_expanded():
                last_node = node
                node = node.select_child()
                if node is None:
                    node = last_node
                    break
        
            # Get the value of the leaf node
            # If the leaf node is a terminal node, the value is one of [-1, 0, +1], otherwise use value network
            value = node.board_state.get_reward() 
            if value is None:
                value = self.value_model.predict(node.board_state)

            # Continue expansion for if the value network returns [-stop_threshold, stop_threshold]
            if (value > -stop_threshold) and (value < stop_threshold):
                # If the game is a draw (from the perspective of the value network), continue expansion
                move_probabilities = self.policy_model.predict(self.board_state) 
                node.expand(move_probabilities)

            node.backpropagate(value)

class Node:
    def __init__(self, board_state, parent, prior):
        self.board_state = board_state
        self.prior = prior
        self.value_sum = 0 # TODO: Handle values for both players, e.g. value network may return -0.7 for the value for black
        self.visit_count = 0
        self.children = {}
        self.parent = parent

    # https://joshvarty.github.io/AlphaZero/
    # https://www.chessprogramming.org/UCT
    def ucb_score(self):
        """
        Calculate the UCB score for the given node.
        """
        return self.value_sum + self.prior * math.sqrt(math.log(self.parent.visit_count + 1, math.e) / self.visit_count + 1)

    def is_expanded(self):
        """
        Check if current node is expanded.
        """
        return len(self.children) > 0
        
    def expand(self, move_probabilities):
        """
        Expand current node based on the possible moves.
        """
        for move, prob in move_probabilities:
            self.children[move] = (Node(self.board_state.step(move), self, prob))

    def select_child(self):
        """
        Select child node with highest UCB score.
        """
        highest_ucb = -math.inf
        selected_child = None

        for child in self.children:
            ucb = child.ucb_score()
            if ucb > highest_ucb:
                highest_ucb = ucb
                selected_child = child
        
        return selected_child

    def backpropagate(self, value):
        """
        Backpropagate the value of the current node up to the root node.
        """
        self.value_sum += value
        self.visit_count += 1

        self.parent.backpropagate()

    def select_move(self, temperature):
        """
        Select move based on the visit distribution of the root node,
        which can be optionally adjusted by a temperature parameter.
        https://arxiv.org/pdf/2012.11045.pdf Section 5.3
        https://arxiv.org/pdf/1905.13521.pdf Section 3.2

        P[a | s] = N(s, a)^{1 / t} / sum_b{N(s, b)^{1 / t}}
        where N(s, a) is the number of times action a was taken in state s
        """
        if (temperature == 0):
            # Select move with highest visit count
            pass
        else:
            # Select move based on visit distribution
            pass
        pass
