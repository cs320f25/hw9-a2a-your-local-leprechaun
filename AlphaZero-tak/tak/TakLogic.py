"""
TakLogic.py - Core game logic for Tak, adapted for AlphaZero
Uses numpy arrays for board representation compatible with neural networks
"""

import numpy as np
import sys
sys.path.append('..')
from game import Game
from flatStone import FlatStone
from standingStone import StandingStone
from capstone import Capstone

class TakLogic:
    """
    Core Tak game logic using numpy arrays for AlphaZero compatibility.

    Board representation:
    - 3D numpy array: (height, n, n) where height = n (max stack size)
    - Each cell stores piece values:
        0 = empty
        1 = white flat
        2 = black flat
        3 = white standing
        4 = black standing
        5 = white capstone
        6 = black capstone
    """

    def __init__(self, n=5):
        self.n = n
        self.piece_counts = self._get_piece_counts(n)

    def _get_piece_counts(self, n):
        """Get standard piece counts for board size."""
        piece_data = {
            3: {"flats": 10, "capstones": 0},
            4: {"flats": 15, "capstones": 0},
            5: {"flats": 21, "capstones": 1},
            6: {"flats": 30, "capstones": 1},
            7: {"flats": 40, "capstones": 1},
            8: {"flats": 50, "capstones": 1}
        }
        return piece_data.get(n, {"flats": n * n, "capstones": 1 if n >= 5 else 0})

    def get_init_board(self):
        """
        Returns initial board state.
        Returns:
            board: (height, n, n) numpy array
            piece_counts: dict with white/black piece counts
            move_number: int, current move (0 = opening)
        """
        board = np.zeros((self.n, self.n, self.n), dtype=np.int8)
        counts = {
            'white_flats': self.piece_counts['flats'],
            'black_flats': self.piece_counts['flats'],
            'white_capstones': self.piece_counts['capstones'],
            'black_capstones': self.piece_counts['capstones']
        }
        return board, counts, 0

    def piece_to_value(self, piece_type, player):
        """Convert piece type and player to board value."""
        # player: 1 = white, -1 = black
        # piece_type: 'flat', 'standing', 'capstone'
        base = 0 if player == 1 else 1
        if piece_type == 'flat':
            return 1 + base
        elif piece_type == 'standing':
            return 3 + base
        elif piece_type == 'capstone':
            return 5 + base
        return 0

    def value_to_piece(self, value):
        """Convert board value to (player, piece_type)."""
        if value == 0:
            return None, None
        if value in [1, 2]:
            return (1 if value == 1 else -1), 'flat'
        if value in [3, 4]:
            return (1 if value == 3 else -1), 'standing'
        if value in [5, 6]:
            return (1 if value == 5 else -1), 'capstone'
        return None, None

    def get_top_piece(self, board, row, col):
        """Get the top piece at a position."""
        for height in range(self.n - 1, -1, -1):
            if board[height, row, col] != 0:
                return self.value_to_piece(board[height, row, col])
        return None, None

    def get_stack_height(self, board, row, col):
        """Get height of stack at position."""
        height = 0
        for h in range(self.n):
            if board[h, row, col] != 0:
                height = h + 1
        return height
