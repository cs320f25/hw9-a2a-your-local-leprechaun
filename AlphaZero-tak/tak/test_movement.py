"""Test movement implementation"""
import numpy as np
import sys
sys.path.append('..')
from TakGame import TakGame

def test_simple_movement():
    """Test simple one-square movement"""
    print("\n" + "="*60)
    print("TEST: Simple Movement")
    print("="*60)

    game = TakGame(5)
    board = game.getInitBoard()
    player = 1

    # Skip opening moves
    board, player = game.getNextState(board, player, 0)  # Move 1
    board, player = game.getNextState(board, player, 1)  # Move 2

    # Place a flat at (2, 2)
    action = 2 * game.n + 2 + 0 * game.n * game.n
    board, next_player = game.getNextState(board, player, action)

    # Place another piece so we can come back to player 1
    board, player = game.getNextState(board, next_player, 5)

    # Now player is back to 1 (white), who owns the piece at (2,2)

    print("Initial board:")
    game.display(board)
    print(f"Current player: {player}")

    # Try to move piece from (2,2) right to (2,3)
    # Movement encoding: position + direction*n² + pattern_idx*4*n²
    # pattern_idx=0 should be (pickup=1, pattern=[1])
    # direction=3 is right (0=up, 1=down, 2=left, 3=right)
    # position = 2*5 + 2 = 12

    movement_start = game.num_placement_actions
    pattern_idx = 0
    direction = 3  # right
    position = 2 * game.n + 2

    action = movement_start + position + direction * game.n * game.n + pattern_idx * 4 * game.n * game.n

    print(f"\nAttempting movement action: {action}")
    print(f"From (2,2) to (2,3), direction=right")

    new_board, next_player = game.getNextState(board, player, action)

    print("\nBoard after movement:")
    game.display(new_board)

    # Check if piece moved
    old_height = game._get_stack_height(board, 2, 2)
    new_height = game._get_stack_height(new_board, 2, 2)
    target_height = game._get_stack_height(new_board, 2, 3)

    print(f"\nHeight at (2,2) before: {old_height}, after: {new_height}")
    print(f"Height at (2,3) after: {target_height}")

    if new_height < old_height and target_height > 0:
        print("[PASS] Movement successful!")
        return True
    else:
        print("[FAIL] Movement did not work")
        return False


def test_capstone_flattening():
    """Test capstone flattening standing stone"""
    print("\n" + "="*60)
    print("TEST: Capstone Flattening")
    print("="*60)

    game = TakGame(5)
    board = game.getInitBoard()
    player = 1

    # Skip opening moves
    board, player = game.getNextState(board, player, 0)
    board, player = game.getNextState(board, player, 1)

    # Place standing stone at (2, 2) by white
    action_wall = 2 * game.n + 2 + 1 * game.n * game.n
    board, player = game.getNextState(board, player, action_wall)

    # Place capstone at (2, 1) by black
    action_cap = 2 * game.n + 1 + 2 * game.n * game.n
    board, next_player = game.getNextState(board, player, action_cap)

    # Place dummy piece so black gets turn again
    board, player = game.getNextState(board, next_player, 5)

    # Now player is black (-1), who owns the capstone at (2,1)

    print("Before movement:")
    game.display(board)

    # Move capstone from (2,1) right to (2,2) - should flatten the wall
    movement_start = game.num_placement_actions
    pattern_idx = 0  # pickup=1, pattern=[1]
    direction = 3  # right
    position = 2 * game.n + 1

    action = movement_start + position + direction * game.n * game.n + pattern_idx * 4 * game.n * game.n

    print(f"\nMoving capstone from (2,1) to (2,2) onto standing stone")

    new_board, next_player = game.getNextState(board, player, action)

    print("\nAfter movement:")
    game.display(new_board)

    # Check if wall was flattened
    height = game._get_stack_height(new_board, 2, 2)
    if height >= 2:
        bottom_piece = new_board[height-2, 2, 2]
        top_piece = new_board[height-1, 2, 2]
        _, bottom_type = game._value_to_piece(bottom_piece)
        _, top_type = game._value_to_piece(top_piece)

        print(f"\nBottom piece type: {bottom_type}")
        print(f"Top piece type: {top_type}")

        if bottom_type == 'flat' and top_type == 'capstone':
            print("[PASS] Standing stone was flattened!")
            return True

    print("[FAIL] Flattening did not work correctly")
    return False


def test_blocked_movement():
    """Test that movement onto capstone is blocked"""
    print("\n" + "="*60)
    print("TEST: Blocked Movement (onto capstone)")
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

    # Now player is black (-1), who owns the flat at (2,1)

    print("Before movement:")
    game.display(board)

    # Try to move flat from (2,1) onto capstone at (2,2) - should fail
    movement_start = game.num_placement_actions
    pattern_idx = 0
    direction = 3  # right
    position = 2 * game.n + 1

    action = movement_start + position + direction * game.n * game.n + pattern_idx * 4 * game.n * game.n

    print(f"\nAttempting to move flat from (2,1) onto capstone at (2,2)")

    new_board, next_player = game.getNextState(board, player, action)

    # Check if board unchanged (move was blocked)
    if np.array_equal(board[:game.n], new_board[:game.n]):
        print("[PASS] Movement correctly blocked!")
        return True
    else:
        print("[FAIL] Movement should have been blocked")
        return False


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# MOVEMENT IMPLEMENTATION TESTS")
    print("#"*60)

    results = []
    results.append(("Simple Movement", test_simple_movement()))
    results.append(("Capstone Flattening", test_capstone_flattening()))
    results.append(("Blocked Movement", test_blocked_movement()))

    print("\n" + "#"*60)
    print("# SUMMARY")
    print("#"*60)
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status}: {name}")
