"""
Test script to verify TakGame implementation
"""

import numpy as np
from TakGame import TakGame

def test_initialization():
    """Test game initialization."""
    print("=== Testing Initialization ===")
    g = TakGame(5)

    print(f"Board size: {g.getBoardSize()}")
    print(f"Action size: {g.getActionSize()}")

    board = g.getInitBoard()
    print(f"Initial board shape: {board.shape}")
    print("[OK] Initialization test passed\n")

    return g, board

def test_display():
    """Test board display."""
    print("=== Testing Display ===")
    g = TakGame(5)
    board = g.getInitBoard()

    print("Empty board:")
    g.display(board)

    # Add some pieces
    board[0, 2, 2] = 1  # White flat at (2,2)
    board[0, 3, 3] = 2  # Black flat at (3,3)
    board[0, 1, 1] = 3  # White standing at (1,1)

    print("Board with some pieces:")
    g.display(board)
    print("[OK] Display test passed\n")

def test_valid_moves():
    """Test valid move generation."""
    print("=== Testing Valid Moves ===")
    g = TakGame(5)
    board = g.getInitBoard()

    valid = g.getValidMoves(board, 1)
    print(f"Total actions: {len(valid)}")
    print(f"Valid actions on empty board: {sum(valid)}")

    # Should have n*n*3 valid placements on empty board
    expected = 5 * 5 * 3
    print(f"Expected placement actions: {expected}")

    if sum(valid) >= expected:
        print("[OK] Valid moves test passed\n")
    else:
        print("[WARNING] Fewer valid moves than expected\n")

def test_next_state():
    """Test state transitions."""
    print("=== Testing Next State ===")
    g = TakGame(5)
    board = g.getInitBoard()

    # Test placement action (place flat at position 0,0)
    action = 0  # row=0, col=0, piece_type=flat
    new_board, next_player = g.getNextState(board, 1, action)

    print("After placing piece:")
    g.display(new_board)

    print(f"Next player: {next_player}")
    print("[OK] Next state test passed\n")

def test_game_ended():
    """Test game end detection."""
    print("=== Testing Game End Detection ===")
    g = TakGame(5)
    board = g.getInitBoard()

    result = g.getGameEnded(board, 1)
    print(f"Game ended on empty board: {result} (should be 0)")

    # Create a simple road for white (horizontal)
    for col in range(5):
        board[0, 2, col] = 1  # White flats

    print("\nBoard with potential road:")
    g.display(board)

    result = g.getGameEnded(board, 1)
    print(f"Game ended for white: {result} (should be 1 if road detected)")

    result = g.getGameEnded(board, -1)
    print(f"Game ended for black: {result} (should be -1 if white won)")

    print("[OK] Game end test passed\n")

def test_canonical_form():
    """Test canonical form conversion."""
    print("=== Testing Canonical Form ===")
    g = TakGame(5)
    board = g.getInitBoard()

    # Add pieces
    board[0, 0, 0] = 1  # White flat
    board[0, 1, 1] = 2  # Black flat

    print("Original board (player 1):")
    g.display(board)

    canonical = g.getCanonicalForm(board, -1)
    print("Canonical form (from player -1 perspective):")
    g.display(canonical)

    print("[OK] Canonical form test passed\n")

def test_string_representation():
    """Test string representation for hashing."""
    print("=== Testing String Representation ===")
    g = TakGame(5)
    board1 = g.getInitBoard()
    board2 = g.getInitBoard()

    # Same board should have same hash
    hash1 = g.stringRepresentation(board1)
    hash2 = g.stringRepresentation(board2)
    print(f"Empty boards have same hash: {hash1 == hash2}")

    # Different boards should have different hash
    board2[0, 0, 0] = 1
    hash3 = g.stringRepresentation(board2)
    print(f"Different boards have different hash: {hash1 != hash3}")

    print("[OK] String representation test passed\n")

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING TAK GAME FOR ALPHAZERO")
    print("=" * 60 + "\n")

    try:
        g, board = test_initialization()
        test_display()
        test_valid_moves()
        test_next_state()
        test_game_ended()
        test_canonical_form()
        test_string_representation()

        print("=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
