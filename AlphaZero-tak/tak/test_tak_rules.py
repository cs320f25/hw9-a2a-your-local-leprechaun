"""
Comprehensive test suite for Tak game rules
Tests all specific Tak rules including:
1. Piece limits (21 flats + 1 capstone per player)
2. Movement (direction + multi-drop)
3. Capstone flattening standing stones
4. Movement restrictions (walls and capstones)
5. Opening rule (first piece is opponent's color)
"""

import numpy as np
import sys
import io
sys.path.append('..')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from TakGame import TakGame


def test_piece_limits():
    """Test that players can only place their number of pieces."""
    print("\n" + "="*60)
    print("TEST 1: Piece Limits Enforcement")
    print("="*60)

    game = TakGame(5)  # 5x5 board: 21 flats + 1 capstone
    board = game.getInitBoard()
    player = 1

    # Test initial piece counts
    max_flats = game.standard_pieces['flats']
    white_flats = int(board[game.n, 0, 0] * max_flats)
    white_caps = int(board[game.n + 1, 0, 0])

    print(f"Initial white pieces: {white_flats} flats, {white_caps} capstone")
    assert white_flats == 21, f"Expected 21 flats, got {white_flats}"
    assert white_caps == 1, f"Expected 1 capstone, got {white_caps}"

    # Place 21 flats
    print("\nPlacing 21 flats...")
    positions = [(i, j) for i in range(5) for j in range(5)][:21]
    for idx, (row, col) in enumerate(positions):
        action = row * game.n + col + 0 * game.n * game.n  # Flat placement
        board, player = game.getNextState(board, player, action)
        player = -player  # Switch back for testing

    # Check piece count after 21 placements
    white_flats = int(board[game.n, 0, 0] * max_flats)
    print(f"After 21 placements: {white_flats} flats remaining")
    assert white_flats == 0, f"Expected 0 flats, got {white_flats}"

    # Try to place 22nd flat - should not be valid
    valid_moves = game.getValidMoves(board, 1)
    flat_actions = [i for i in range(game.n * game.n)]  # First n² actions are flat placements
    valid_flat_actions = [a for a in flat_actions if valid_moves[a] == 1]

    print(f"Valid flat placement actions after 21 placements: {len(valid_flat_actions)}")
    assert len(valid_flat_actions) == 0, "Should not be able to place more flats"

    print("✓ Piece limits enforced correctly!")
    return True


def test_opening_rule():
    """Test that first piece placed should be of opponent's color."""
    print("\n" + "="*60)
    print("TEST 2: Opening Rule (First Piece is Opponent's Color)")
    print("="*60)

    game = TakGame(5)
    board = game.getInitBoard()
    player = 1  # White player moves first

    # First move: White places a flat at (0, 0)
    action = 0 * game.n + 0 + 0 * game.n * game.n  # row=0, col=0, flat
    new_board, next_player = game.getNextState(board, player, action)

    # Check what piece was placed
    piece_value = new_board[0, 0, 0]  # First layer, position (0,0)

    print(f"Player {player} placed piece with value: {piece_value}")
    print(f"Expected: {2} (opponent's flat) for opening rule")

    # According to opening rule, first piece should be opponent's color
    # White player (1) should place black flat (2)
    expected_value = 2  # Black flat

    if piece_value == expected_value:
        print("✓ Opening rule implemented correctly!")
        return True
    else:
        print(f"✗ Opening rule NOT implemented: placed {piece_value}, expected {expected_value}")
        return False


def test_movement_basic():
    """Test basic movement in one direction."""
    print("\n" + "="*60)
    print("TEST 3: Basic Movement Implementation")
    print("="*60)

    game = TakGame(5)
    board = game.getInitBoard()
    player = 1

    # Skip opening moves
    board, player = game.getNextState(board, player, 0)  # Move 1
    board, player = game.getNextState(board, player, 1)  # Move 2

    # Place a flat at (2, 2) for white
    action = 2 * game.n + 2 + 0 * game.n * game.n
    board, next_player = game.getNextState(board, player, action)

    # Place another piece so white can move
    board, player = game.getNextState(board, next_player, 5)

    print("Initial board:")
    game.display(board)
    print(f"Current player: {player}")

    # Try to move piece from (2,2) to (2,3) - one square right
    movement_start_idx = game.num_placement_actions
    print(f"\nMovement actions start at index: {movement_start_idx}")
    print(f"Total actions: {game.action_size}")
    print(f"Number of movement actions: {game.num_movement_actions}")

    # Try a movement action: pattern_idx=0, direction=3 (right), position=12
    if game.num_movement_actions > 0:
        pattern_idx = 0
        direction = 3  # right
        position = 2 * game.n + 2
        test_movement_action = movement_start_idx + position + direction * game.n * game.n + pattern_idx * 4 * game.n * game.n

        print(f"Attempting movement from (2,2) to (2,3)")
        new_board, _ = game.getNextState(board, player, test_movement_action)

        # Check if piece moved
        old_height = game._get_stack_height(board, 2, 2)
        new_height = game._get_stack_height(new_board, 2, 2)
        target_height = game._get_stack_height(new_board, 2, 3)

        if new_height < old_height and target_height > 0:
            print("✓ Movement implementation works!")
            return True
        else:
            print("✗ Movement NOT implemented correctly")
            return False
    else:
        print("✗ No movement actions encoded")
        return False


def test_capstone_flattening():
    """Test that capstone flattens standing stones when moving onto them."""
    print("\n" + "="*60)
    print("TEST 4: Capstone Flattening Standing Stones")
    print("="*60)

    game = TakGame(5)
    board = game.getInitBoard()
    player = 1

    # Skip opening moves
    board, player = game.getNextState(board, player, 0)
    board, player = game.getNextState(board, player, 1)

    # Place standing stone at (2, 2) by white
    action_standing = 2 * game.n + 2 + 1 * game.n * game.n
    board, player = game.getNextState(board, player, action_standing)

    # Place capstone at (2, 1) by black
    action_capstone = 2 * game.n + 1 + 2 * game.n * game.n
    board, next_player = game.getNextState(board, player, action_capstone)

    # Place dummy to get black turn again
    board, player = game.getNextState(board, next_player, 5)

    print("Before movement:")
    game.display(board)

    # Move capstone from (2,1) to (2,2) - should flatten the wall
    movement_start = game.num_placement_actions
    pattern_idx = 0
    direction = 3  # right
    position = 2 * game.n + 1

    action = movement_start + position + direction * game.n * game.n + pattern_idx * 4 * game.n * game.n

    print(f"\nMoving capstone from (2,1) onto standing stone at (2,2)")

    new_board, _ = game.getNextState(board, player, action)

    print("After movement:")
    game.display(new_board)

    # Check if wall was flattened
    height = game._get_stack_height(new_board, 2, 2)
    if height >= 2:
        bottom_piece = new_board[height-2, 2, 2]
        _, bottom_type = game._value_to_piece(bottom_piece)

        if bottom_type == 'flat':
            print("✓ Standing stone was flattened!")
            return True
        else:
            print(f"✗ Bottom piece is {bottom_type}, expected flat")
            return False

    print("✗ Flattening did not work")
    return False


def test_movement_restrictions():
    """Test that pieces can't move onto walls or capstones."""
    print("\n" + "="*60)
    print("TEST 5: Movement Restrictions")
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

    print("Board setup: Flat at (2,1), Capstone at (2,2)")
    game.display(board)

    # Try to move flat onto capstone - should be blocked
    movement_start = game.num_placement_actions
    pattern_idx = 0
    direction = 3  # right
    position = 2 * game.n + 1

    action = movement_start + position + direction * game.n * game.n + pattern_idx * 4 * game.n * game.n

    print(f"\nAttempting to move flat from (2,1) onto capstone at (2,2)")

    new_board, _ = game.getNextState(board, player, action)

    # Check if board unchanged (move was blocked)
    if np.array_equal(board[:game.n], new_board[:game.n]):
        print("✓ Movement correctly blocked!")
        return True
    else:
        print("✗ Movement should have been blocked")
        return False


def test_multi_drop_movement():
    """Test movement with multiple drops."""
    print("\n" + "="*60)
    print("TEST 6: Multi-Drop Movement")
    print("="*60)

    game = TakGame(5)
    board = game.getInitBoard()
    player = 1

    # Skip opening moves
    board, player = game.getNextState(board, player, 0)
    board, player = game.getNextState(board, player, 1)

    # Build a 3-high stack at (2, 2) with white pieces
    # Place first flat
    action = 2 * game.n + 2 + 0 * game.n * game.n
    board, player = game.getNextState(board, player, action)

    # Have black place somewhere else
    board, player = game.getNextState(board, player, 5)

    # White places another flat at (2, 2)
    board, player = game.getNextState(board, player, 2 * game.n + 2)

    # Black places elsewhere
    board, player = game.getNextState(board, player, 6)

    # White places third flat at (2, 2)
    board, player = game.getNextState(board, player, 2 * game.n + 2)

    # Black places elsewhere to give white turn
    board, player = game.getNextState(board, player, 7)

    print("Initial board with 3-high stack at (2, 2):")
    game.display(board)

    # Move 2 pieces from (2,2) right, dropping [1, 1]
    # Find pattern for pickup=2, drops=[1, 1]
    pattern_idx = 1  # Pattern 1 is (pickup=2, drops=[1, 1])
    direction = 3  # right
    position = 2 * game.n + 2
    movement_start = game.num_placement_actions

    action = movement_start + position + direction * game.n * game.n + pattern_idx * 4 * game.n * game.n

    print(f"\nMoving 2 pieces from (2,2) with pattern [1, 1]")

    new_board, _ = game.getNextState(board, player, action)

    print("After movement:")
    game.display(new_board)

    # Check results
    source_height = game._get_stack_height(new_board, 2, 2)
    target1_height = game._get_stack_height(new_board, 2, 3)
    target2_height = game._get_stack_height(new_board, 2, 4)

    print(f"\nStack heights - Source (2,2): {source_height}, (2,3): {target1_height}, (2,4): {target2_height}")

    if source_height == 1 and target1_height == 1 and target2_height == 1:
        print("✓ Multi-drop movement works!")
        return True
    else:
        print("✗ Multi-drop movement failed")
        return False


def run_all_tests():
    """Run all Tak rule tests."""
    print("\n" + "#"*60)
    print("# TAK GAME RULES - COMPREHENSIVE TEST SUITE")
    print("#"*60)

    results = []

    # Test 1: Piece limits
    try:
        results.append(("Piece Limits", test_piece_limits()))
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        results.append(("Piece Limits", False))

    # Test 2: Opening rule
    try:
        results.append(("Opening Rule", test_opening_rule()))
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        results.append(("Opening Rule", False))

    # Test 3: Basic movement
    try:
        results.append(("Movement", test_movement_basic()))
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        results.append(("Movement", False))

    # Test 4: Capstone flattening
    try:
        results.append(("Capstone Flattening", test_capstone_flattening()))
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        results.append(("Capstone Flattening", False))

    # Test 5: Movement restrictions
    try:
        results.append(("Movement Restrictions", test_movement_restrictions()))
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        results.append(("Movement Restrictions", False))

    # Test 6: Multi-drop movement
    try:
        results.append(("Multi-Drop Movement", test_multi_drop_movement()))
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        results.append(("Multi-Drop Movement", False))

    # Summary
    print("\n" + "#"*60)
    print("# TEST SUMMARY")
    print("#"*60)
    for test_name, result in results:
        if result is True:
            print(f"✓ {test_name}: PASSED")
        elif result is False:
            print(f"✗ {test_name}: FAILED")
        else:
            print(f"⊘ {test_name}: NEEDS IMPLEMENTATION")

    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    needs_impl = sum(1 for _, r in results if r is None)

    print(f"\nResults: {passed} passed, {failed} failed, {needs_impl} need implementation")


if __name__ == "__main__":
    run_all_tests()
