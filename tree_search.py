import math
import torch
from chess_env import *
from typing import Callable, Any, Dict
import copy

num_simulations = 5
stop_threshold = 0.1

class MCTS:
    def __init__(
        self,
        initial_state: ChessEnv,
        model: torch.nn.Module,
        # environment: Any,
        state_to_model_input: Callable[[Any], torch.Tensor],
        policy_to_move_probabilities: Callable[[torch.Tensor], Dict[Any, float]]
    ):
        self.root = Node(initial_state, 1, None, 0)
        self.current_node = self.root
        self.model = model
        # self.environment = environment
        self.state_to_model_input = state_to_model_input
        self.policy_to_move_probabilities = policy_to_move_probabilities


    def search(self):
        """
        MCTS algorithm, select, expand, backpropagate.
        """
        policy = self.model(self.state_to_model_input(self.current_node.board_state).unsqueeze(0))[1]
        move_probabilities = self.policy_to_move_probabilities(policy)

        self.current_node.expand(move_probabilities)

        for _ in range(num_simulations):
            node = self.current_node

            # Select a leaf node
            while node.is_expanded():
                last_node = node
                node = node.select_child()
                if node is None:
                    node = last_node
                    break
            value, policy = self.model(self.state_to_model_input(node.board_state).unsqueeze(0))

            # Continue expansion for if the value network returns [-stop_threshold, stop_threshold]
            if (value > -stop_threshold) and (value < stop_threshold):
                # If the game is a draw (from the perspective of the value network), continue expansion
                move_probabilities = self.policy_to_move_probabilities(policy)
                node.expand(move_probabilities)

            node.backpropagate(value, node.player)
    
    def update_state(self, move: Any):
        """
        Update the board state based on the given move.
        """
        self.current_node = self.current_node.children[move]
        

    def get_root(self):
        """
        Return the root node.
        """
        return self.root
    
    def get_current_node(self):
        """
        Return the current node.
        """
        return self.current_node


class Node:
    def __init__(self, board_state, player, parent, prior):
        self.board_state = board_state
        self.player = player
        self.prior = prior
        self.value_sum = 0
        self.visit_count = 0
        self.children = {}
        self.parent = parent
        self.reward = 0

    # https://joshvarty.github.io/AlphaZero/
    # https://www.chessprogramming.org/UCT
    def ucb_score(self):
        """
        Calculate the UCB score for the given node.
        """
        return self.value_sum + self.prior * math.sqrt(math.log(self.parent.visit_count + 1, math.e) / (self.visit_count + 1))


    def is_expanded(self):
        """
        Check if current node is expanded.
        """
        return len(self.children) > 0
        

    def expand(self, move_probabilities):
        """
        Expand current node based on the possible moves.
        """
        board = self.board_state

        for possible_move in self.board_state.get_possible_moves():
            board.push(possible_move)
            self.children[possible_move] = (Node(copy.deepcopy(board), self.player * - 1, self, move_probabilities[possible_move]))
            board.pop()


    def select_child(self):
        """
        Select child node with highest UCB score.
        """
        highest_ucb = -math.inf
        selected_child = None

        for child in self.children.values():
            ucb = child.ucb_score()
            if ucb > highest_ucb:
                highest_ucb = ucb
                selected_child = child
        
        return selected_child


    def backpropagate(self, value, player):
        """
        Backpropagate the value of the current node up to the root node.
        """
        if (self.player == player):
            self.value_sum += value
        else:
            self.value_sum -= player

        self.visit_count += 1

        if self.parent:
            self.parent.backpropagate(value, player * -1)


    def select_move(self):
        """
        Select move based on the visit distribution of the root node,
        which can be optionally adjusted by a temperature parameter.
        https://arxiv.org/pdf/2012.11045.pdf Section 5.3
        https://arxiv.org/pdf/1905.13521.pdf Section 3.2

        P[a | s] = N(s, a)^{1 / t} / sum_b{N(s, b)^{1 / t}}
        where N(s, a) is the number of times action a was taken in state s
        """
        # Select action a with highest visit count
        visit_counts = [child.visit_count for child in self.children.values()]
        moves = [move for move in self.children.keys()]
        max_visit_count = max(visit_counts)
        index = visit_counts.index(max_visit_count)
        return moves[index]

    def get_move_probabilities(self):
        """
        Return the move probabilities based on the visit counts of the root node.
        """
        visit_counts = {move: child.visit_count for move, child in self.children.items()}
        total_visit_count = sum(visit_counts.values())
        move_probabilities = {move: visit_count / total_visit_count for move, visit_count in visit_counts.items()}
        return move_probabilities
