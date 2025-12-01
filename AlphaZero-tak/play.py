"""
Play against your trained Tak agent!
"""

import numpy as np
from Arena import Arena
from MCTS import MCTS
from tak.TakGame import TakGame
from tak.TakNNet import NNetWrapper as nn
from utils import dotdict

# Load trained model
def load_agent(board_size=5):
    """Load a trained agent from checkpoint."""
    g = TakGame(board_size)
    nnet = nn(g)

    try:
        nnet.load_checkpoint('./temp/', 'best.pth.tar')
        print("Loaded trained model successfully!")
    except:
        print("Warning: Could not load trained model. Using untrained network.")

    return g, nnet

# Human player
class HumanPlayer:
    def __init__(self, game):
        self.game = game

    def __call__(self, board):
        """
        Get action from human player.
        board is in canonical form (always playing as player 1)
        """
        self.game.display(board)
        valid_moves = self.game.getValidMoves(board, 1)

        print("\nValid actions:")
        valid_actions = [i for i, v in enumerate(valid_moves) if v == 1]

        # Show first 20 valid moves for readability
        print(f"Available actions: {valid_actions[:20]}")
        if len(valid_actions) > 20:
            print(f"... and {len(valid_actions) - 20} more")

        # Placement action format
        print(f"\nPlacement actions (0-{3*self.game.n*self.game.n - 1}):")
        print(f"  Format: row * {self.game.n} + col + piece_type * {self.game.n*self.game.n}")
        print(f"  piece_type: 0=flat, 1=standing, 2=capstone")
        print(f"  Example: To place flat at (2,3): {2 * self.game.n + 3}")

        while True:
            try:
                action = int(input(f"\nEnter action (0-{self.game.getActionSize()-1}): "))

                if action < 0 or action >= len(valid_moves):
                    print(f"Invalid action range! Must be 0-{self.game.getActionSize()-1}")
                    continue

                if valid_moves[action] == 0:
                    print("That move is not valid! Try again.")
                    continue

                return action

            except ValueError:
                print("Please enter a valid number!")
            except KeyboardInterrupt:
                print("\nGame cancelled.")
                exit(0)

def play_game(board_size=5, num_mcts_sims=25, human_first=True):
    """
    Play a game against the trained agent.

    Args:
        board_size: Size of the board (3, 4, or 5)
        num_mcts_sims: Number of MCTS simulations for AI (higher = stronger)
        human_first: If True, human plays first; otherwise AI plays first
    """
    print(f"\n{'='*50}")
    print(f"Playing Tak {board_size}x{board_size}")
    print(f"AI strength: {num_mcts_sims} MCTS simulations")
    print(f"{'='*50}\n")

    # Load game and agent
    game, nnet = load_agent(board_size)

    # Create MCTS for agent
    args = dotdict({'numMCTSSims': num_mcts_sims, 'cpuct': 1.0})
    mcts = MCTS(game, nnet, args)

    # Agent player function
    def agent_player(board):
        return np.argmax(mcts.getActionProb(board, temp=0))

    # Human player
    human = HumanPlayer(game)

    # Create arena
    if human_first:
        print("You are playing as White (w)")
        arena = Arena(human, agent_player, game, display=game.display)
    else:
        print("You are playing as Black (b)")
        arena = Arena(agent_player, human, game, display=game.display)

    # Play game
    print("\nStarting game...\n")
    winner = arena.playGame(verbose=True)

    print("\n" + "="*50)
    if winner == 1:
        print("Player 1 (White) wins!")
    elif winner == -1:
        print("Player 2 (Black) wins!")
    else:
        print("Draw!")
    print("="*50 + "\n")

if __name__ == "__main__":
    # Configuration
    BOARD_SIZE = 5          # 3, 4, or 5
    MCTS_SIMS = 10          # AI strength (10=easy, 25=medium, 100=hard)
    HUMAN_FIRST = True      # Set to False to let AI go first

    play_game(board_size=BOARD_SIZE, num_mcts_sims=MCTS_SIMS, human_first=HUMAN_FIRST)
