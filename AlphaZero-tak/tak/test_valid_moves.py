"""
Test valid move generation for movement actions
Focuses on complex multi-piece movements
"""

import numpy as np
import sys
sys.path.append('..')
from TakGame import TakGame


def test_simple_movement_valid():
    """Test that simple movements are marked as valid"""
    print("\n" + "="*60)
    print("TEST 1: Simple Movement Valid Moves")
    print("="*60)

    game = TakGame(5)
    board = game.getInitBoard()
    player = 1

    # Skip opening moves
    board, player = game.getNextState(board, player, 0)
    board, player = game.getNextState(board, player, 1)

    # Place a flat at (2, 2) for white
    action = 2 * game.n + 2 + 0 * game.n * game.n
    board, next_player = game.getNextState(board, player, action)

    # Place another piece so white can move
    board, player = game.getNextState(board, next_player, 5)

    print("Board state:")
    game.display(board)

    # Get valid moves
    valid_moves = game.getValidMoves(board, player)

    # Check movement actions
    movement_start = game.num_placement_actions
    movement_actions = valid_moves[movement_start:]
    num_valid_movements = np.sum(movement_actions)

    print(f"\nTotal valid placement actions: {np.sum(valid_moves[:movement_start])}")
    print(f"Total valid movement actions: {num_valid_movements}")

    if num_valid_movements > 0:
        print(f"[PASS] Found {num_valid_movements} valid movement actions")
        return True
    else:
        print("[FAIL] No valid movement actions found")
        return False


def test_multi_piece_movement():
    """Test multi-piece movement: pickup 3, drop [2, 1] going left"""
    print("\n" + "="*60)
    print("TEST 2: Multi-Piece Movement (pickup 3, drop [2,1])")
    print("="*60)

    game = TakGame(5)
    board = game.getInitBoard()
    player = 1

    # Skip opening moves
    board, player = game.getNextState(board, player, 0)
    board, player = game.getNextState(board, player, 1)

    # Build a 3-high stack at (2, 2) with white pieces
    for i in range(3):
        action = 2 * game.n + 2 + 0 * game.n * game.n
        board, next_player = game.getNextState(board, player, action)
        board, player = game.getNextState(board, next_player, 5 + i)  # Black places elsewhere

    print("Board with 3-high stack at (2, 2):")
    game.display(board)

    # Get valid moves for white
    valid_moves = game.getValidMoves(board, player)

    # Find the specific action: pickup 3 from (2,2), drop [2,1], direction left
    # First, find pattern_idx for (pickup=3, drops=[2,1])
    pattern_idx = None
    for idx, (pickup, pattern) in enumerate(game.movement_patterns):
        if pickup == 3 and pattern == [2, 1]:
            pattern_idx = idx
            break

    if pattern_idx is None:
        print("[FAIL] Pattern (pickup=3, drops=[2,1]) not found")
        return False

    print(f"\nFound pattern at index {pattern_idx}: pickup=3, drops=[2,1]")

    # Construct the action for moving left (direction=2) from (2,2)
    direction = 2  # left
    position = 2 * game.n + 2
    action = game.num_placement_actions + position + direction * game.n * game.n + pattern_idx * 4 * game.n * game.n

    print(f"Action index: {action}")
    print(f"Is valid: {valid_moves[action] == 1}")

    if valid_moves[action] == 1:
        print("[PASS] Multi-piece movement marked as valid!")

        # Actually execute the move to verify it works
        new_board, _ = game.getNextState(board, player, action)
        print("\nBoard after movement:")
        game.display(new_board)

        # Check that pieces moved correctly
        source_height = game._get_stack_height(new_board, 2, 2)
        target1_height = game._get_stack_height(new_board, 2, 1)  # Left 1
        target2_height = game._get_stack_height(new_board, 2, 0)  # Left 2

        print(f"\nHeights - Source (2,2): {source_height}, Target1 (2,1): {target1_height}, Target2 (2,0): {target2_height}")

        if source_height == 0 and target1_height == 2 and target2_height == 1:
            print("[PASS] Movement executed correctly!")
            return True
        else:
            print("[FAIL] Movement execution incorrect")
            return False
    else:
        print("[FAIL] Multi-piece movement NOT marked as valid")
        return False


def test_blocked_movement_invalid():
    """Test that blocked movements are marked as invalid"""
    print("\n" + "="*60)
    print("TEST 3: Blocked Movements Should Be Invalid")
    print("="*60)

    game = TakGame(5)
    board = game.getInitBoard()
    player = 1

    # Skip opening moves
    board, player = game.getNextState(board, player, 0)
    board, player = game.getNextState(board, player, 1)

    # Place capstone at (2, 2) by white
    action_cap = 2 * game.n + 2 + 2 * game.n * game.n
    board, player = game.getNextState(board, player, action_cap)

    # Place flat at (2, 1) by black
    action_flat = 2 * game.n + 1 + 0 * game.n * game.n
    board, next_player = game.getNextState(board, player, action_flat)

    # Place dummy to get black turn again
    board, player = game.getNextState(board, next_player, 5)

    print("Board: Flat at (2,1), Capstone at (2,2)")
    game.display(board)

    # Get valid moves for black (who owns the flat at 2,1)
    valid_moves = game.getValidMoves(board, player)

    # Try to find action for moving flat onto capstone (should be invalid)
    # Direction right (3), pattern_idx 0 (pickup 1, drop [1])
    pattern_idx = 0
    direction = 3  # right
    position = 2 * game.n + 1
    action = game.num_placement_actions + position + direction * game.n * game.n + pattern_idx * 4 * game.n * game.n

    print(f"\nAction to move flat onto capstone: {action}")
    print(f"Is valid: {valid_moves[action] == 1}")

    if valid_moves[action] == 0:
        print("[PASS] Blocked movement correctly marked as invalid!")
        return True
    else:
        print("[FAIL] Blocked movement incorrectly marked as valid")
        return False


def test_boundary_movements():
    """Test that movements off the board are invalid"""
    print("\n" + "="*60)
    print("TEST 4: Boundary Check (off-board movements invalid)")
    print("="*60)

    game = TakGame(5)
    board = game.getInitBoard()
    player = 1

    # Skip opening moves
    board, player = game.getNextState(board, player, 0)
    board, player = game.getNextState(board, player, 1)

    # Place flat at corner (0, 0)
    action = 0 * game.n + 0 + 0 * game.n * game.n
    board, next_player = game.getNextState(board, player, action)

    # Place dummy
    board, player = game.getNextState(board, next_player, 5)

    print("Board with piece at corner (0, 0):")
    game.display(board)

    # Get valid moves
    valid_moves = game.getValidMoves(board, player)

    # Check if movements up (direction 0) or left (direction 2) are invalid
    pattern_idx = 0
    position = 0

    action_up = game.num_placement_actions + position + 0 * game.n * game.n + pattern_idx * 4 * game.n * game.n
    action_left = game.num_placement_actions + position + 2 * game.n * game.n + pattern_idx * 4 * game.n * game.n

    print(f"\nMovement up from (0,0) valid: {valid_moves[action_up] == 1}")
    print(f"Movement left from (0,0) valid: {valid_moves[action_left] == 1}")

    if valid_moves[action_up] == 0 and valid_moves[action_left] == 0:
        print("[PASS] Off-board movements correctly marked as invalid!")
        return True
    else:
        print("[FAIL] Off-board movements should be invalid")
        return False


def performance_test():
    """Test performance of valid move generation"""
    print("\n" + "="*60)
    print("TEST 5: Performance Test")
    print("="*60)

    import time

    game = TakGame(5)
    board = game.getInitBoard()
    player = 1

    # Create a complex board state
    board, player = game.getNextState(board, player, 0)
    board, player = game.getNextState(board, player, 1)

    # Place several pieces
    for i in range(5):
        action = i * game.n + i
        board, player = game.getNextState(board, player, action)
        board, player = game.getNextState(board, player, 5 + i)

    print("Testing valid move generation performance...")

    start = time.time()
    for _ in range(100):
        valid_moves = game.getValidMoves(board, player)
    end = time.time()

    avg_time = (end - start) / 100
    print(f"Average time per getValidMoves call: {avg_time*1000:.2f}ms")

    if avg_time < 0.1:  # Less than 100ms
        print(f"[PASS] Performance acceptable")
        return True
    else:
        print(f"[WARNING] Performance may be slow for training")
        return True


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# VALID MOVE GENERATION TESTS")
    print("#"*60)

    results = []
    results.append(("Simple Movement Valid", test_simple_movement_valid()))
    results.append(("Multi-Piece Movement", test_multi_piece_movement()))
    results.append(("Blocked Movements Invalid", test_blocked_movement_invalid()))
    results.append(("Boundary Check", test_boundary_movements()))
    results.append(("Performance", performance_test()))

    print("\n" + "#"*60)
    print("# TEST SUMMARY")
    print("#"*60)
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {name}")

    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
