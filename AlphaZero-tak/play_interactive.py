"""
Interactive play against your trained Tak agent!
User-friendly interface with coordinate-based input.
"""

import numpy as np
from Arena import Arena
from MCTS import MCTS
from tak.TakGame import TakGame
from tak.TakNNet import NNetWrapper as nn
from utils import dotdict
import os


class InteractiveHumanPlayer:
    """Human player with user-friendly input."""

    def __init__(self, game):
        self.game = game

    def __call__(self, board):
        """Get action from human player."""
        self.game.display(board)
        valid_moves = self.game.getValidMoves(board, 1)

        print("\n" + "="*50)
        print("YOUR TURN")
        print("="*50)

        while True:
            try:
                print("\nEnter your move:")
                print("  Format: <piece> <row> <col>")
                print("  Pieces: f=flat, s=standing, c=capstone")
                print("  Example: 'f 2 3' to place flat at row 2, col 3")
                print("  Type 'quit' to exit")

                user_input = input("\n> ").strip().lower()

                if user_input in ['quit', 'exit', 'q']:
                    print("\nGame ended by user.")
                    exit(0)

                # Parse input
                parts = user_input.split()
                if len(parts) != 3:
                    print("❌ Invalid format! Use: <piece> <row> <col>")
                    continue

                piece_char, row_str, col_str = parts

                # Map piece type
                piece_map = {'f': 0, 'flat': 0, 's': 1, 'standing': 1, 'c': 2, 'cap': 2, 'capstone': 2}
                if piece_char not in piece_map:
                    print(f"❌ Invalid piece '{piece_char}'! Use f, s, or c")
                    continue

                piece_type = piece_map[piece_char]
                row = int(row_str)
                col = int(col_str)

                # Validate coordinates
                if row < 0 or row >= self.game.n or col < 0 or col >= self.game.n:
                    print(f"❌ Invalid coordinates! Must be 0-{self.game.n-1}")
                    continue

                # Calculate action
                action = row * self.game.n + col + piece_type * self.game.n * self.game.n

                # Check if valid
                if action < 0 or action >= len(valid_moves):
                    print(f"❌ Action out of range!")
                    continue

                if valid_moves[action] == 0:
                    print("❌ That move is not valid! Square may be occupied.")
                    continue

                piece_names = ['flat', 'standing stone', 'capstone']
                print(f"✓ Placing {piece_names[piece_type]} at ({row}, {col})")
                return action

            except ValueError:
                print("❌ Invalid input! Make sure row and col are numbers.")
            except KeyboardInterrupt:
                print("\n\nGame cancelled.")
                exit(0)
            except Exception as e:
                print(f"❌ Error: {e}")


def play_game(model_path='./temp/', model_file='best.pth.tar',
              board_size=5, num_mcts_sims=10, human_first=True):
    """
    Play an interactive game against the trained agent.

    Args:
        model_path: Path to model directory
        model_file: Model filename
        board_size: Size of the board
        num_mcts_sims: AI strength (higher = stronger but slower)
        human_first: If True, human plays first as White
    """
    print("\n" + "="*50)
    print(f"🎮  TAK {board_size}x{board_size} - HUMAN vs AI")
    print("="*50)
    print(f"AI Strength: {num_mcts_sims} MCTS simulations")

    # Load game
    game = TakGame(board_size)

    # Load trained model
    nnet = nn(game)

    full_path = os.path.join(model_path, model_file)
    if os.path.exists(full_path):
        nnet.load_checkpoint(model_path, model_file)
        print(f"✓ Loaded model: {full_path}")
    else:
        print(f"⚠️  Warning: Model not found at {full_path}")
        print("Using untrained network (will play randomly)")

    # Create AI player
    args = dotdict({'numMCTSSims': num_mcts_sims, 'cpuct': 1.0})
    mcts = MCTS(game, nnet, args)

    def ai_player(board):
        print("\n🤖 AI is thinking...")
        action = np.argmax(mcts.getActionProb(board, temp=0))

        # Decode action for display
        if action < game.num_placement_actions:
            piece_type_idx = action // (game.n * game.n)
            pos = action % (game.n * game.n)
            row = pos // game.n
            col = pos % game.n
            piece_names = ['flat', 'standing stone', 'capstone']
            print(f"🤖 AI places {piece_names[piece_type_idx]} at ({row}, {col})")
        else:
            print(f"🤖 AI makes move action {action}")

        return action

    # Create human player
    human = InteractiveHumanPlayer(game)

    # Setup game
    if human_first:
        print("\n👤 You are playing as White (w) - You go first!")
        print("Goal: Connect opposite edges with a road of your pieces")
        player1, player2 = human, ai_player
    else:
        print("\n👤 You are playing as Black (b) - AI goes first!")
        print("Goal: Connect opposite edges with a road of your pieces")
        player1, player2 = ai_player, human

    # Play the game
    print("\n" + "="*50)
    print("GAME START")
    print("="*50 + "\n")

    arena = Arena(player1, player2, game, display=game.display)
    winner = arena.playGame(verbose=False)

    # Show final board
    board = game.getInitBoard()
    print("\n" + "="*50)
    print("GAME OVER")
    print("="*50)
    game.display(board)

    if winner == 1:
        if human_first:
            print("🎉 You win! Congratulations!")
        else:
            print("🤖 AI wins!")
    elif winner == -1:
        if human_first:
            print("🤖 AI wins!")
        else:
            print("🎉 You win! Congratulations!")
    else:
        print("🤝 Draw!")
    print("="*50 + "\n")


if __name__ == "__main__":
    # ========== CONFIGURATION ==========

    # Model settings
    MODEL_PATH = './models/5x5_easy_v1/'  # or './temp/'
    MODEL_FILE = 'best.pth.tar'

    # Game settings
    BOARD_SIZE = 5                    # Must match your trained model
    AI_STRENGTH = 10                  # MCTS sims: 10=easy, 25=medium, 50+=hard
    HUMAN_GOES_FIRST = True          # Set False to let AI go first

    # ===================================

    play_game(
        model_path=MODEL_PATH,
        model_file=MODEL_FILE,
        board_size=BOARD_SIZE,
        num_mcts_sims=AI_STRENGTH,
        human_first=HUMAN_GOES_FIRST
    )
