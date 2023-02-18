from typing import List, Tuple
import numpy as np


class Env():
    def __init__(self, state: np.array = None):
        """
        Initialize the environment.
        :param fen: standard fen chess position, string
        """
        self.state = state
        if state is None:
            self.reset()
        self.white_to_move = True

    def reset(self, state: Tuple[np.array] = None, white_to_move: bool = True):
        """
        Reset the environment to the given state. If state is None, reset to the starting position.
        :param state:
        :return:
        """
        if state is None:
            self.state = np.zeros((2, 6, 7), dtype=bool)
        else:
            self.state = state
        self.white_to_move = white_to_move

    def step(self, col: int):
        """
        Make a move in the environment.
        :param col: the column to drop the piece in
        :return:
        """

        if col not in self.get_possible_moves():  # check if the move is legal
            raise Exception("The move is not in legal_moves")
        else:
            row = np.where((self.state[0][:, col] == 0) & (self.state[1][:, col] == 0))[0][0]

            if self.white_to_move:
                self.state[0][row, col] = True
            else:
                self.state[1][row, col] = True
            self.white_to_move = not self.white_to_move

    def get_possible_moves(self):  # returns list of items in the form: Move.from_uci('a2a3')
        """
        Get all legal moves from the current state.
        :return:
        """
        possible_moves = []
        for i in range(7):
            if not self.state[0][-1, i] and not self.state[1][-1, i]:
                possible_moves.append(i)
        return possible_moves

    def get_state(self):
        """
        Get the current state of the environment.
        :return:
        """
        return self.state.copy(), self.white_to_move

    def get_state_after_move(self, col: int):
        """
        Get the state of the environment after a move.
        :param col: the column to drop the piece in
        :return:
        """
        initial_state = self.get_state()
        self.step(col)
        new_state = self.get_state()
        self.reset(*initial_state)
        return new_state

    def get_reward(self, verbose=False):
        """
        Written pretty much completely by GitHub Copilot
        return 1 if win for white, 0 for draw, -1 for loss for white, None if game is not over
        :return:
        """
        # check for horizontal win
        for i in range(6):
            for j in range(4):
                if self.state[0][i, j] and self.state[0][i, j + 1] and self.state[0][i, j + 2] and self.state[0][
                        i, j + 3]:
                    if verbose:
                        print('horizontal win for white')
                    return 1
                if self.state[1][i, j] and self.state[1][i, j + 1] and self.state[1][i, j + 2] and self.state[1][
                        i, j + 3]:
                    if verbose:
                        print('horizontal win for black')
                    return -1
        # check for vertical win
        for i in range(3):
            for j in range(7):
                if self.state[0][i, j] and self.state[0][i + 1, j] and self.state[0][i + 2, j] and self.state[0][
                        i + 3, j]:
                    if verbose:
                        print('vertical win for white')
                    return 1
                if self.state[1][i, j] and self.state[1][i + 1, j] and self.state[1][i + 2, j] and self.state[1][
                    i + 3, j]:
                    if verbose:
                        print('vertical win for black')
                    return -1
        # check for diagonal win
        for i in range(3):
            for j in range(4):
                if self.state[0][i, j] and self.state[0][i + 1, j + 1] and self.state[0][i + 2, j + 2] and \
                        self.state[0][i + 3, j + 3]:
                    if verbose:
                        print('diagonal1 win for white')
                    return 1
                if self.state[1][i, j] and self.state[1][i + 1, j + 1] and self.state[1][i + 2, j + 2] and \
                        self.state[1][i + 3, j + 3]:
                    if verbose:
                        print('diagonal1 win for black')
                    return -1
        for i in range(3):
            for j in range(4):
                if self.state[0][i + 3, j] and self.state[0][i + 2, j + 1] and self.state[0][i + 1, j + 2] and \
                        self.state[0][i, j + 3]:
                    if verbose:
                        print('diagonal2 win for white')
                    return 1
                if self.state[1][i + 3, j] and self.state[1][i + 2, j + 1] and self.state[1][i + 1, j + 2] and \
                        self.state[1][i, j + 3]:
                    if verbose:
                        print('diagonal2 win for black')
                    return -1
        if len(self.get_possible_moves()) == 0:
            return 0
        return None

    def display(self):
        """
        Display the current state of the environment.
        :return:
        """
        print(' - - - - - - -')
        for i in range(6)[::-1]:
            for j in range(7):
                if self.state[0][i, j]:
                    print('|X', end='')
                elif self.state[1][i, j]:
                    print('|O', end='')
                else:
                    print('| ', end='')
            print('|')
            print(' - - - - - - -')


def predict(model, state: Tuple[np.array], white_to_move: bool):
    """
    Get the model's prediction for the given state.
    :param model: the model to use
    :param state: the state to predict
    :param white_to_move: whether it is white's turn to move
    :return:
    """
    if white_to_move:
        state = (state[0], state[1])
    else:
        state = (state[1], state[0])
    state = np.array(state).reshape(1, 2, 6, 7)
    return model.predict(state)[0]


def policy_predictions_to_move_probabilities(policy_predictions: np.array, legal_moves: List[int]):
    """
    Convert the model's policy predictions to a probability distribution over the legal moves.
    :param policy_predictions: the model's policy predictions
    :param legal_moves: the legal moves
    :return:
    """
    move_probabilities = np.zeros(7)
    for i in range(7):
        if i in legal_moves:
            move_probabilities[i] = policy_predictions[i]
    return zip(range(7), move_probabilities)
