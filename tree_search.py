
import math
from chess_env import *

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

        for _ in range(1000):
            node = root
            
            # Select a leaf node
            while node.is_expanded():
                last_node = node
                node = node.select_child()
                if node is None:
                    node = last_node
                    break
        
            # Get the value of the leaf node
            value = self.value_model.predict(node.board_state)

            # TODO: 
            if (value == 0):
                # If the game is a draw (from the perspective of the value network), continue expansion
                move_probabilities = self.policy_model.predict(self.board_state) 
                node.expand(move_probabilities)

            node.backpropagate(value)

class Node:
    def __init__(self, board_state, parent, prior):
        self.board_state = board_state
        self.prior = prior
        self.value_sum = 0
        self.visit_count = 0
        self.children = {}
        self.parent = parent

    def is_expanded(self):
        """
        Check if current node is expanded.
        """
        return len(self.children) > 0

    # https://joshvarty.github.io/AlphaZero/
    # https://www.chessprogramming.org/UCT
    def ucb_score(self):
        """
        Calculate the UCB score for the given node.
        """
        return self.value_sum + self.prior * math.sqrt(math.log(self.parent.visit_count, math.e) / self.visit_count) 
        
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
