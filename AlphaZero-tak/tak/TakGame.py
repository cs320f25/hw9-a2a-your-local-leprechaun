"""
TakGame.py - AlphaZero-compatible interface for Tak
Implements the Game class interface required by alpha-zero-general
"""

import numpy as np
from itertools import product
import sys
sys.path.append('..')

class TakGame:
    """
    Tak game implementation for AlphaZero.

    Game state represented as tuple: (board, piece_counts, move_number, player)
    - board: (height, n, n) numpy array
    - piece_counts: dict with piece availability
    - move_number: int (0-1 = opening, 2+ = normal play)
    - player: 1 (white) or -1 (black)
    """

    def __init__(self, n=5):
        self.n = n
        self._init_piece_counts(n)
        self._init_action_encoding()

    def _init_piece_counts(self, n):
        """Initialize piece counts based on board size."""
        piece_data = {
            3: {"flats": 10, "capstones": 0},
            4: {"flats": 15, "capstones": 0},
            5: {"flats": 21, "capstones": 1},
            6: {"flats": 30, "capstones": 1},
            7: {"flats": 40, "capstones": 1},
            8: {"flats": 50, "capstones": 1}
        }
        self.standard_pieces = piece_data.get(n, {"flats": n * n, "capstones": 1 if n >= 5 else 0})

    def _init_action_encoding(self):
        """
        Initialize action encoding scheme.

        Actions are encoded as integers:
        - Placements: 0 to 3n²-1
          Format: row * n + col + piece_type * n²
          piece_type: 0=flat, 1=standing, 2=capstone

        - Movements: 3n² onwards
          Format: position + direction*n² + pickup_config*4*n²
          direction: 0=up, 1=down, 2=left, 3=right
        """
        n = self.n

        # Placement actions: 3n²
        self.num_placement_actions = 3 * n * n

        # Movement actions: enumerate all pickup/drop patterns
        self.movement_patterns = []
        for pickup in range(1, n + 1):
            # Generate all ways to partition 'pickup' pieces
            patterns = self._generate_drop_patterns(pickup, pickup)
            self.movement_patterns.extend([(pickup, pattern) for pattern in patterns])

        # Each square, each direction, each pattern
        self.num_movement_actions = n * n * 4 * len(self.movement_patterns)

        self.action_size = self.num_placement_actions + self.num_movement_actions

    def _generate_drop_patterns(self, total, max_drops):
        """
        Generate all valid drop patterns for picking up 'total' pieces.
        max_drops is the maximum number of squares we can drop over.
        """
        if total == 1:
            return [[1]]

        patterns = []

        def generate(remaining, current_pattern, max_remaining_drops):
            if remaining == 0:
                patterns.append(current_pattern[:])
                return
            if max_remaining_drops == 0:
                return

            # Try dropping different amounts at current position
            for drop_amount in range(1, remaining + 1):
                current_pattern.append(drop_amount)
                generate(remaining - drop_amount, current_pattern, max_remaining_drops - 1)
                current_pattern.pop()

        generate(total, [], max_drops)
        return patterns

    def getInitBoard(self):
        """
        Returns:
            board: Initial board state (height, n, n) numpy array
        """
        # Board is (height, n, n) with all zeros
        board = np.zeros((self.n, self.n, self.n), dtype=np.int8)

        # Encode piece counts and move number in extra channels for neural network
        # For now, just return the board; piece counts will be tracked separately
        return board

    def getBoardSize(self):
        """
        Returns:
            (height, n, n): board dimensions for neural network
        """
        return (self.n, self.n, self.n)

    def getActionSize(self):
        """
        Returns:
            int: Total number of possible actions
        """
        return self.action_size

    def getNextState(self, board, player, action):
        """
        Apply action and return new state.

        Args:
            board: Current board state
            player: 1 or -1
            action: Integer action index

        Returns:
            (board, next_player): New board state and next player to move
        """
        # Create a copy of the board
        new_board = np.copy(board)

        # Decode and apply action
        # For now, implement basic placement logic
        # TODO: Implement full action decoding and application

        if action < self.num_placement_actions:
            # Placement action
            piece_type_idx = action // (self.n * self.n)
            pos = action % (self.n * self.n)
            row = pos // self.n
            col = pos % self.n

            # Find first empty height level
            height = self._get_stack_height(new_board, row, col)

            if height < self.n:
                # Map piece type
                piece_types = ['flat', 'standing', 'capstone']
                piece_type = piece_types[piece_type_idx]

                # Encode piece value
                value = self._piece_to_value(piece_type, player)
                new_board[height, row, col] = value

        else:
            # Movement action
            # TODO: Implement movement logic
            pass

        return new_board, -player

    def getValidMoves(self, board, player):
        """
        Returns:
            valid_moves: Binary numpy array of shape (action_size,)
                        where 1 = valid move, 0 = invalid
        """
        valid = np.zeros(self.action_size, dtype=np.int8)

        # TODO: Implement full move validation
        # For now, mark all placements on empty squares as valid

        # Check placement actions
        for piece_type_idx in range(3):
            for row in range(self.n):
                for col in range(self.n):
                    height = self._get_stack_height(board, row, col)
                    if height == 0:  # Empty square
                        action = row * self.n + col + piece_type_idx * self.n * self.n
                        valid[action] = 1

        return valid

    def getGameEnded(self, board, player):
        """
        Check if game has ended.

        Returns:
            0: Game continues
            1: Current player won
            -1: Current player lost
            0.0001: Draw (small non-zero value)
        """
        # TODO: Implement road checking and flat counting
        # For now, return 0 (game continues)

        # Check for road win for both players
        if self._check_road(board, player):
            return 1
        if self._check_road(board, -player):
            return -1

        # Check if board is full
        if self._is_board_full(board):
            # Count flats
            score = self._count_flats(board, player) - self._count_flats(board, -player)
            if score > 0:
                return 1
            elif score < 0:
                return -1
            else:
                return 0.0001  # Draw

        return 0

    def getCanonicalForm(self, board, player):
        """
        Return board from perspective of player.

        For player -1, flip the board representation so it appears as if they are player 1.
        """
        if player == 1:
            return board
        else:
            # Flip piece ownership
            canonical = np.copy(board)
            # Swap white pieces (1,3,5) with black pieces (2,4,6)
            mask_white_flat = (canonical == 1)
            mask_black_flat = (canonical == 2)
            canonical[mask_white_flat] = 2
            canonical[mask_black_flat] = 1

            mask_white_standing = (canonical == 3)
            mask_black_standing = (canonical == 4)
            canonical[mask_white_standing] = 4
            canonical[mask_black_standing] = 3

            mask_white_cap = (canonical == 5)
            mask_black_cap = (canonical == 6)
            canonical[mask_white_cap] = 6
            canonical[mask_black_cap] = 5

            return canonical

    def getSymmetries(self, board, pi):
        """
        Return list of (board, pi) pairs for all symmetries.

        For Tak, we have 8 symmetries: 4 rotations × 2 reflections
        However, actions need to be transformed accordingly.

        For simplicity, we'll start with no symmetries and add later.
        """
        return [(board, pi)]

    def stringRepresentation(self, board):
        """
        Return string representation of board for hashing.
        """
        return board.tobytes()

    # Helper methods

    def _get_stack_height(self, board, row, col):
        """Get height of stack at position."""
        for h in range(self.n):
            if board[h, row, col] == 0:
                return h
        return self.n

    def _piece_to_value(self, piece_type, player):
        """Convert piece type and player to board value."""
        # player: 1 = white, -1 = black
        base = 0 if player == 1 else 1
        if piece_type == 'flat':
            return 1 + base
        elif piece_type == 'standing':
            return 3 + base
        elif piece_type == 'capstone':
            return 5 + base
        return 0

    def _check_road(self, board, player):
        """
        Check if player has a winning road using BFS.
        A road connects two opposite edges with flats/capstones.
        """
        target_flat = 1 if player == 1 else 2
        target_cap = 5 if player == 1 else 6

        # Check horizontal roads (left to right)
        for row in range(self.n):
            if self._has_road_from(board, row, 0, target_flat, target_cap, "horizontal"):
                return True

        # Check vertical roads (top to bottom)
        for col in range(self.n):
            if self._has_road_from(board, 0, col, target_flat, target_cap, "vertical"):
                return True

        return False

    def _has_road_from(self, board, start_row, start_col, target_flat, target_cap, direction):
        """Use BFS to find road from starting position to opposite edge."""
        # Check if starting position is valid
        height = self._get_stack_height(board, start_row, start_col)
        if height == 0:
            return False

        top = board[height - 1, start_row, start_col]
        if top != target_flat and top != target_cap:
            return False

        # BFS
        visited = set()
        queue = [(start_row, start_col)]
        visited.add((start_row, start_col))

        while queue:
            row, col = queue.pop(0)

            # Check if reached opposite edge
            if direction == "horizontal" and col == self.n - 1:
                return True
            if direction == "vertical" and row == self.n - 1:
                return True

            # Check all 4 adjacent positions
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_row, new_col = row + dr, col + dc

                if (new_row, new_col) in visited:
                    continue

                if new_row < 0 or new_row >= self.n or new_col < 0 or new_col >= self.n:
                    continue

                # Check if this position is part of the road
                height = self._get_stack_height(board, new_row, new_col)
                if height > 0:
                    top = board[height - 1, new_row, new_col]
                    if top == target_flat or top == target_cap:
                        visited.add((new_row, new_col))
                        queue.append((new_row, new_col))

        return False

    def _is_board_full(self, board):
        """Check if board is full."""
        for row in range(self.n):
            for col in range(self.n):
                if self._get_stack_height(board, row, col) == 0:
                    return False
        return True

    def _count_flats(self, board, player):
        """Count flats controlled by player."""
        count = 0
        target_flat = 1 if player == 1 else 2
        target_cap = 5 if player == 1 else 6

        for row in range(self.n):
            for col in range(self.n):
                height = self._get_stack_height(board, row, col)
                if height > 0:
                    top = board[height - 1, row, col]
                    if top == target_flat or top == target_cap:
                        count += 1
        return count

    def display(self, board):
        """Display board in human-readable format."""
        print("\n   ", end="")
        for i in range(self.n):
            print(f" {i}  ", end="")
        print()

        for row in range(self.n):
            print(f"{row}  ", end="")
            for col in range(self.n):
                height = self._get_stack_height(board, row, col)
                if height == 0:
                    print("[  ]", end="")
                else:
                    top = board[height - 1, row, col]
                    player, piece_type = self._value_to_piece(top)
                    color = 'w' if player == 1 else 'b'

                    if piece_type == 'flat':
                        print(f"[{color}{height}]", end="")
                    elif piece_type == 'standing':
                        print(f"[{color}S]", end="")
                    elif piece_type == 'capstone':
                        print(f"[{color}C]", end="")
            print()
        print()

    def _value_to_piece(self, value):
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
