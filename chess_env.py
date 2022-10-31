import chess


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
        Get the current state of the environment.
        :return:
        """
        pass

    def get_reward(self):
        """
        return 1 if win for white, 0 for draw, -1 for loss for white, None if game is not over
        :return:
        """
        pass


def state_to_alpha_zero_input(state):
    """
    Convert the state to the format expected by the AlphaZero model.
    :param state:
    :return:
    """
    pass
