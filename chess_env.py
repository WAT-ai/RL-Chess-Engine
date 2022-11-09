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

    def step(self, move): # move param needs to be in the following format: move = chess.Move.from_uci("g1f3")
        """
        Make a move in the environment.
        :param move:
        :return:
        """

        if move not in self.legal_moves: #check if the move is legal
            raise Exception("The move is not in legal_moves")
        else:
            self.push(move)

        return None

    def get_possible_moves(self): # returns list of items in the form: Move.from_uci('a2a3')
        """
        Get all legal moves from the current state.
        :return:
        """
        possible_moves = list(self.legal_moves)
        return possible_moves

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
