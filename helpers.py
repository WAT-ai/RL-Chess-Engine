from chess import SQUARES, Move, KNIGHT, BISHOP, ROOK, QUEEN
import numpy as np

MOVE_PLANES = [
    # up
    8, 16, 24, 32, 40, 48, 56,
    # down
    -8, -16, -24, -32, -40, -48, -56,
    # left
    -1, -2, -3, -4, -5, -6, -7,
    # right
    1, 2, 3, 4, 5, 6, 7,
    # up-left
    7, 14, 21, 28, 35, 42, 49,
    # up-right
    9, 18, 27, 36, 45, 54, 63,
    # down-left
    -9, -18, -27, -36, -45, -54, -63,
    # down-right
    -7, -14, -21, -28, -35, -42, -49,
    # knight moves
    17, 15, 10, 6, -17, -15, -10, -6,
]

UNDERPROMOTION_MOVE_PLANES = [
    # up
    8,
    # up-left
    7,
    # up-right
    9
]

PROMOTION_PIECES = [KNIGHT, BISHOP, ROOK, QUEEN]

planes = []
for trip in MOVE_PLANES:
    plane = []
    up_plane = trip // 8
    right_plane = trip % 8
    for square in SQUARES:
        up_square = square // 8
        right_square = square % 8
        if 0 <= (up_square + up_plane) < 8 and 0 <= (right_square + right_plane) < 8:
            plane.append(Move(square, square + trip))
        else:
            plane.append(Move(square, 100))
    planes.append(plane)

for trip in UNDERPROMOTION_MOVE_PLANES:
    for piece in PROMOTION_PIECES:
        plane = []
        up_plane = trip // 8
        right_plane = trip % 8
        for square in SQUARES:
            up_square = square // 8
            right_square = square % 8
            if 0 <= (up_square + up_plane) < 8 and 0 <= (right_square + right_plane) < 8:
                plane.append(Move(square, square + trip, promotion=piece))
            else:
                plane.append(Move(square, 100))
        planes.append(plane)

moves = np.array(planes).reshape((73, 8, 8))


def policy_to_move_probabilities(policy: np.ndarray):
    """
    Convert the policy output of the neural network to a dictionary of move probabilities.
    """
    move_probabilities = {m: p for m, p in zip(moves.flatten(), policy.flatten())}
    return move_probabilities
