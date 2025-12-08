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
            board: Initial board state with piece counts encoded

        Board structure (depth, n, n):
        - Layers 0 to n-1: The actual game board (height levels)
        - Layer n: Player 1 (white) remaining flats (normalized 0-1)
        - Layer n+1: Player 1 remaining capstones (0 or 1)
        - Layer n+2: Player 2 (black) remaining flats (normalized 0-1)
        - Layer n+3: Player 2 remaining capstones (0 or 1)
        """
        # Board has n layers for game state + 4 layers for piece counts
        board = np.zeros((self.n + 4, self.n, self.n), dtype=np.float32)

        # Initialize piece counts in extra layers (fill entire layer with same value)
        # Normalize flats to 0-1 range
        max_flats = self.standard_pieces['flats']
        board[self.n, :, :] = max_flats / max_flats  # Player 1 flats (1.0)
        board[self.n + 1, :, :] = self.standard_pieces['capstones']  # Player 1 capstones
        board[self.n + 2, :, :] = max_flats / max_flats  # Player 2 flats (1.0)
        board[self.n + 3, :, :] = self.standard_pieces['capstones']  # Player 2 capstones

        return board

    def getBoardSize(self):
        """
        Returns:
            (depth, n, n): board dimensions for neural network
            depth = n (game layers) + 4 (piece count layers)
        """
        return (self.n + 4, self.n, self.n)

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
            board: Current board state (includes piece counts in extra layers)
            player: 1 or -1
            action: Integer action index

        Returns:
            (board, next_player): New board state and next player to move
        """
        # Create a copy of the board
        new_board = np.copy(board)

        # Decode and apply action
        if action < self.num_placement_actions:
            # Placement action
            piece_type_idx = action // (self.n * self.n)
            pos = action % (self.n * self.n)
            row = pos // self.n
            col = pos % self.n

            # Find first empty height level (only check game layers)
            height = self._get_stack_height(new_board, row, col)

            if height < self.n:
                # Map piece type
                piece_types = ['flat', 'standing', 'capstone']
                piece_type = piece_types[piece_type_idx]

                # Check if this is an opening move (first 2 pieces)
                # Count total pieces on board
                total_pieces = 0
                for r in range(self.n):
                    for c in range(self.n):
                        if self._get_stack_height(new_board, r, c) > 0:
                            total_pieces += 1

                is_opening_move = total_pieces < 2

                # For opening moves, place opponent's piece (only flats allowed)
                if is_opening_move:
                    piece_type = 'flat'  # Opening moves must be flats
                    value = self._piece_to_value(piece_type, -player)  # Opponent's piece
                else:
                    # Normal move: place own piece
                    value = self._piece_to_value(piece_type, player)
                new_board[height, row, col] = value

                # Decrement piece count in the appropriate layer
                max_flats = self.standard_pieces['flats']
                if player == 1:
                    if piece_type == 'capstone':
                        new_board[self.n + 1, :, :] -= 1
                    else:
                        # Decrement and normalize flats
                        current_flats = new_board[self.n, 0, 0] * max_flats
                        new_board[self.n, :, :] = (current_flats - 1) / max_flats
                else:  # player == -1
                    if piece_type == 'capstone':
                        new_board[self.n + 3, :, :] -= 1
                    else:
                        current_flats = new_board[self.n + 2, 0, 0] * max_flats
                        new_board[self.n + 2, :, :] = (current_flats - 1) / max_flats

        else:
            # Movement action
            action_offset = action - self.num_placement_actions

            # Decode movement action
            # Format: position + direction*n² + pattern_idx*4*n²
            n = self.n
            pattern_idx = action_offset // (4 * n * n)
            remainder = action_offset % (4 * n * n)
            direction = remainder // (n * n)
            position = remainder % (n * n)

            start_row = position // n
            start_col = position % n

            # Validate pattern index
            if pattern_idx >= len(self.movement_patterns):
                return new_board, -player  # Invalid action

            pickup_count, drop_pattern = self.movement_patterns[pattern_idx]

            # Get direction vector
            direction_vectors = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
            dr, dc = direction_vectors[direction]

            # Pick up pieces from starting stack
            start_height = self._get_stack_height(new_board, start_row, start_col)

            # Validate pickup
            if start_height < pickup_count:
                return new_board, -player  # Can't pick up more than stack height

            # Check if player controls the top piece
            if start_height > 0:
                top_piece = new_board[start_height - 1, start_row, start_col]
                top_player, _ = self._value_to_piece(top_piece)
                if top_player != player:
                    return new_board, -player  # Can't move opponent's stack

            # Pick up pieces (from top of stack)
            carried_pieces = []
            for i in range(pickup_count):
                height_idx = start_height - pickup_count + i
                if height_idx >= 0 and height_idx < self.n:
                    carried_pieces.append(new_board[height_idx, start_row, start_col])
                    new_board[height_idx, start_row, start_col] = 0

            # Move and drop pieces according to pattern
            current_row, current_col = start_row, start_col
            carried_idx = 0

            for drop_count in drop_pattern:
                # Move to next square
                current_row += dr
                current_col += dc

                # Check bounds
                if current_row < 0 or current_row >= n or current_col < 0 or current_col >= n:
                    # Invalid move - restore board and return
                    return np.copy(board), -player

                # Check if we can drop here
                target_height = self._get_stack_height(new_board, current_row, current_col)

                if target_height > 0:
                    # Check top piece of target square
                    top_piece = new_board[target_height - 1, current_row, current_col]
                    top_player, top_type = self._value_to_piece(top_piece)

                    # Check if move is blocked
                    # Can't move onto capstones
                    if top_type == 'capstone':
                        return np.copy(board), -player

                    # Can't move onto standing stones unless we're a capstone
                    if top_type == 'standing':
                        # Check if we're moving with a capstone on top
                        # The last piece we're dropping is the top of our stack
                        is_last_drop = (drop_count == drop_pattern[-1] and
                                      carried_idx + drop_count == len(carried_pieces))

                        if is_last_drop:
                            # Check if top carried piece is a capstone
                            top_carried = carried_pieces[carried_idx + drop_count - 1]
                            _, carried_type = self._value_to_piece(top_carried)

                            if carried_type == 'capstone':
                                # Flatten the standing stone
                                player_color = top_player
                                new_board[target_height - 1, current_row, current_col] = \
                                    self._piece_to_value('flat', player_color)
                            else:
                                # Can't move onto standing stone
                                return np.copy(board), -player
                        else:
                            # Not the last drop, can't move onto standing stone
                            return np.copy(board), -player

                # Drop pieces
                for i in range(drop_count):
                    if carried_idx < len(carried_pieces):
                        drop_height = self._get_stack_height(new_board, current_row, current_col)
                        if drop_height < n:
                            new_board[drop_height, current_row, current_col] = carried_pieces[carried_idx]
                            carried_idx += 1

        return new_board, -player

    def _is_valid_movement(self, board, start_row, start_col, direction, pickup_count, drop_pattern):
        """
        Check if a movement is valid.

        Args:
            board: Current board state
            start_row, start_col: Starting position
            direction: 0=up, 1=down, 2=left, 3=right
            pickup_count: Number of pieces to pick up
            drop_pattern: List of drop counts (e.g., [2, 1])

        Returns:
            bool: True if movement is valid
        """
        n = self.n
        direction_vectors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dr, dc = direction_vectors[direction]

        # Get the pieces we're carrying
        start_height = self._get_stack_height(board, start_row, start_col)
        carried_pieces = []
        for i in range(pickup_count):
            height_idx = start_height - pickup_count + i
            if height_idx >= 0 and height_idx < n:
                carried_pieces.append(board[height_idx, start_row, start_col])

        # Simulate the movement
        current_row, current_col = start_row, start_col
        carried_idx = 0

        for drop_idx, drop_count in enumerate(drop_pattern):
            # Move to next square
            current_row += dr
            current_col += dc

            # Check bounds
            if current_row < 0 or current_row >= n or current_col < 0 or current_col >= n:
                return False  # Would go off board

            # Check if we can drop here
            target_height = self._get_stack_height(board, current_row, current_col)

            if target_height > 0:
                # Check top piece of target square
                top_piece = board[target_height - 1, current_row, current_col]
                _, top_type = self._value_to_piece(top_piece)

                # Can't move onto capstones
                if top_type == 'capstone':
                    return False

                # Can't move onto standing stones unless we're a capstone
                if top_type == 'standing':
                    # Check if we're moving with a capstone on top of our stack
                    # The piece that will end up on top is the last one we're dropping
                    is_last_drop = (drop_idx == len(drop_pattern) - 1)

                    if is_last_drop and carried_idx + drop_count <= len(carried_pieces):
                        # Check if the top carried piece is a capstone
                        top_carried = carried_pieces[carried_idx + drop_count - 1]
                        _, carried_type = self._value_to_piece(top_carried)

                        if carried_type != 'capstone':
                            return False  # Can't flatten wall without capstone
                    else:
                        return False  # Not the last drop, can't move onto wall

            # Check if we have enough pieces to drop
            if carried_idx + drop_count > len(carried_pieces):
                return False

            carried_idx += drop_count

        # Make sure we dropped all pieces
        if carried_idx != len(carried_pieces):
            return False

        return True

    def getValidMoves(self, board, player):
        """
        Returns:
            valid_moves: Binary numpy array of shape (action_size,)
                        where 1 = valid move, 0 = invalid

        Enforces piece count limits and placement rules.
        """
        valid = np.zeros(self.action_size, dtype=np.int8)

        # Get remaining pieces for current player
        max_flats = self.standard_pieces['flats']
        if player == 1:
            remaining_flats = board[self.n, 0, 0] * max_flats
            remaining_capstones = board[self.n + 1, 0, 0]
        else:
            remaining_flats = board[self.n + 2, 0, 0] * max_flats
            remaining_capstones = board[self.n + 3, 0, 0]

        # Check placement actions
        for piece_type_idx in range(3):
            piece_types = ['flat', 'standing', 'capstone']
            piece_type = piece_types[piece_type_idx]

            # Check if player has this piece type available
            # Use >= 1.0 to avoid floating point precision issues
            has_piece = False
            if piece_type in ['flat', 'standing']:
                has_piece = remaining_flats >= 1.0
            elif piece_type == 'capstone':
                has_piece = remaining_capstones >= 0.5

            if not has_piece:
                continue  # Skip this piece type

            for row in range(self.n):
                for col in range(self.n):
                    height = self._get_stack_height(board, row, col)
                    if height == 0:  # Empty square
                        action = row * self.n + col + piece_type_idx * self.n * self.n
                        valid[action] = 1

        # Check movement actions
        # This is computationally expensive but necessary for proper gameplay
        for action_offset in range(self.num_movement_actions):
            action = self.num_placement_actions + action_offset

            # Decode movement action
            n = self.n
            pattern_idx = action_offset // (4 * n * n)
            remainder = action_offset % (4 * n * n)
            direction = remainder // (n * n)
            position = remainder % (n * n)

            start_row = position // n
            start_col = position % n

            # Validate pattern index
            if pattern_idx >= len(self.movement_patterns):
                continue

            pickup_count, drop_pattern = self.movement_patterns[pattern_idx]

            # Check if there's a stack at this position
            start_height = self._get_stack_height(board, start_row, start_col)
            if start_height == 0:
                continue  # No pieces to move

            # Check if we can pick up this many pieces
            if pickup_count > start_height:
                continue

            # In Tak, carry limit is the board size (can't pick up more than n pieces)
            if pickup_count > n:
                continue

            # Check if player controls the top piece
            top_piece = board[start_height - 1, start_row, start_col]
            top_player, _ = self._value_to_piece(top_piece)
            if top_player != player:
                continue  # Can't move opponent's stack

            # Validate the movement path
            if self._is_valid_movement(board, start_row, start_col, direction, pickup_count, drop_pattern):
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
        This includes swapping piece ownership AND piece count layers.
        """
        if player == 1:
            return board
        else:
            # Flip piece ownership in game layers
            canonical = np.copy(board)

            # Swap white pieces (1,3,5) with black pieces (2,4,6) in game layers
            for h in range(self.n):
                mask_white_flat = (canonical[h] == 1)
                mask_black_flat = (canonical[h] == 2)
                canonical[h][mask_white_flat] = 2
                canonical[h][mask_black_flat] = 1

                mask_white_standing = (canonical[h] == 3)
                mask_black_standing = (canonical[h] == 4)
                canonical[h][mask_white_standing] = 4
                canonical[h][mask_black_standing] = 3

                mask_white_cap = (canonical[h] == 5)
                mask_black_cap = (canonical[h] == 6)
                canonical[h][mask_white_cap] = 6
                canonical[h][mask_black_cap] = 5

            # Swap piece count layers (player 1 <-> player 2)
            temp_flats = canonical[self.n].copy()
            temp_caps = canonical[self.n + 1].copy()
            canonical[self.n] = canonical[self.n + 2]
            canonical[self.n + 1] = canonical[self.n + 3]
            canonical[self.n + 2] = temp_flats
            canonical[self.n + 3] = temp_caps

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
        """Display board in human-readable format with piece counts."""
        # Display piece counts
        max_flats = self.standard_pieces['flats']
        white_flats = int(board[self.n, 0, 0] * max_flats)
        white_caps = int(board[self.n + 1, 0, 0])
        black_flats = int(board[self.n + 2, 0, 0] * max_flats)
        black_caps = int(board[self.n + 3, 0, 0])

        print("\n" + "="*40)
        print(f"White (w): {white_flats} flats, {white_caps} capstone(s)")
        print(f"Black (b): {black_flats} flats, {black_caps} capstone(s)")
        print("="*40)

        # Display board
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
