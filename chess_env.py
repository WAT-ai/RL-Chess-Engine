import chess
import numpy as np


class ChessEnv(chess.Board):

    def __init__(self, fen: str = None):
        """
        Initialize the environment.
        :param fen: standard fen chess position, string
        """
        super().__init__(fen=fen)

    def reset(self, state=None):
        """
        Reset the environment to the given state. If state is None, reset to the starting position.
        :param state:
        :return:
        """
        if state == None:
            self.reset()
        else:
            if(isinstance(state, str)):
                super().__init__(fen=state)
        pass

    def step(self, move):
        """
        Make a move in the environment.
        :param move:
        :return:
        """
        pass

    def get_possible_moves(self):
        """
        Get all legal moves from the current state.
        :return:
        """
        pass

    def get_state(self):
        """
        Get the current state of the environment. (FEN)
        :return:
        """
        return self.fen()

    def get_reward(self):
        """
        return 1 if win for white, 0 for draw, -1 for loss for white, None if game is not over
        :return:
        """
        if self.is_game_over:
            # self.outcome().winner is true if white wins, false if black wins and none if it's a draw
            return 1 if self.outcome().winner else (-1 if self.outcome().winner is False else 0)
        return None

def state_to_alpha_zero_input(state):
    """
    Convert the state to the format expected by the AlphaZero model.
    :param state: 
    :return:
    """
    
    pass

    
