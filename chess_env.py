import chess
import numpy as np
import torch


class ChessEnv(chess.Board):

    def __init__(self, fen: str = None):
        """
        Initialize the environment.
        :param fen: standard fen chess position, string
        """
        if fen is None:
            super().__init__()
        else:
            super().__init__(fen=fen)

    def reset(self, state=None):
        """
        Reset the environment to the given state. If state is None, reset to the starting position.
        :param state:
        :return:
        """
        if state is None:
            super().reset()
        else:
            if(isinstance(state, str)):
                super().__init__(fen=state)
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
        Get the current state of the environment. (FEN)
        :return:
        """
        return self.fen()

    def get_reward(self):
        """
        return 1 if win for white, 0 for draw, -1 for loss for white, None if game is not over
        :return:
        """
        if self.is_game_over():
            # self.result() = '1-0' if white wins, '0-1' if black wins, '1/2-1/2' if draw
            white_result = self.result().split('-')[0]
            return 1 if white_result == '1' else (-1 if white_result == '0' else 0)
        return None


def state_to_alpha_zero_input(state):
    """
    Convert the state to the format expected by the AlphaZero model.
    :param state: 
    :return:
    """
    # state is of type chess.board
    if(isinstance(state, chess.Board)):
        # initially it was 8 by 8 by 19, but the last two matrices are used for repition checking, but those values can be represented as a boolean
        array = np.zeros((8, 8, 19), dtype=int)

        for square, piece in state.piece_map().items():
            rank, file = chess.square_rank(square), chess.square_file(square)
            piece_type, color = piece.piece_type, piece.color

            # The first six planes encode the pieces of the active player,
            # the following six those of the active player's opponent. Since
            # this class always stores boards oriented towards the white player,
            # White is considered to be the active player here.
            offset = 0 if color == chess.WHITE else 6

            # Chess enumerates piece types beginning with one, which we have
            # to account for
            idx = piece_type - 1

            array[rank, file, idx + offset] = 1

        array[:, :, 12] = state.is_repetition(3)
        array[:, :, 13] = state.has_kingside_castling_rights(color=chess.WHITE)
        array[:, :, 14] = state.has_queenside_castling_rights(color=chess.WHITE)
        array[:, :, 15] = state.has_kingside_castling_rights(color=chess.BLACK)
        array[:, :, 16] = state.has_queenside_castling_rights(color=chess.BLACK)
        array[:, :, 17] = state.halfmove_clock
        array[:, :, 18] = state.turn
        

        # Repetition counters  #is_repetition returns true if this board position was repeated 2 or 3 times. If repitition occurs 3 times, it would be an automatic draw. Also, reptition two times is not needed
        # False gets converted into a zero matrix
        #array[:, :, 12] = state.is_repetition(2)
        #array[:, :, 13] = state.is_repetition(3)

        # instead of 8 19 by 8 matrices, we get 19 8 by 8 matrices due to the transpose
        # castling_rights represented
        # for turn, True is white's turn, and False is black's turn
        return torch.Tensor(array.transpose(2, 0, 1))
