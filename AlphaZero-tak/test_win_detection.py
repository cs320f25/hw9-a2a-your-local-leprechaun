"""Test win detection"""
import numpy as np
from tak.TakGame import TakGame

game = TakGame(5)
board = game.getInitBoard()

# Create a horizontal road for White (player 1) from left to right at row 2
# White flat = 1
for col in range(5):
    board[0, 2, col] = 1  # Place white flats at row 2, all columns

print("Board with horizontal white road at row 2:")
game.display(board)

# Check if White won
result_white = game.getGameEnded(board, 1)
result_black = game.getGameEnded(board, -1)

print(f"\ngetGameEnded(board, player=1 [White]): {result_white}")
print(f"getGameEnded(board, player=-1 [Black]): {result_black}")

if result_white == 1:
    print("✓ White win detected correctly from White's perspective!")
elif result_black == -1:
    print("✓ White win detected correctly from Black's perspective!")
else:
    print("✗ Win NOT detected! Bug in road checking.")

# Also test vertical road
board2 = game.getInitBoard()
for row in range(5):
    board2[0, row, 2] = 2  # Place black flats at column 2, all rows

print("\n\nBoard with vertical black road at column 2:")
game.display(board2)

result_white2 = game.getGameEnded(board2, 1)
result_black2 = game.getGameEnded(board2, -1)

print(f"\ngetGameEnded(board, player=1 [White]): {result_white2}")
print(f"getGameEnded(board, player=-1 [Black]): {result_black2}")

if result_black2 == 1:
    print("✓ Black win detected correctly from Black's perspective!")
elif result_white2 == -1:
    print("✓ Black win detected correctly from White's perspective!")
else:
    print("✗ Win NOT detected! Bug in road checking.")
