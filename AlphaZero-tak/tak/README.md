# Tak for AlphaZero General

This folder contains a Tak implementation compatible with the [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) framework.

## Structure

```
alpha_zero/
├── TakGame.py          # Main game interface for AlphaZero
├── TakLogic.py         # Core game logic helpers
├── TakNNet.py          # Neural network wrapper
├── pytorch/
│   ├── NNet.py         # PyTorch neural network architecture
│   └── __init__.py
├── test_tak_game.py    # Test suite
└── README.md           # This file
```

## Installation

1. Clone the alpha-zero-general repository:
```bash
git clone https://github.com/suragnair/alpha-zero-general.git
cd alpha-zero-general
```

2. Copy this `alpha_zero/` folder into the alpha-zero-general directory and rename to `tak/`:
```bash
cp -r /path/to/tak-game-python/alpha_zero ./tak
```

3. Install dependencies:
```bash
pip install torch numpy
```

## Usage

### Training

Create a training script (e.g., `main_tak.py`):

```python
from Coach import Coach
from tak.TakGame import TakGame as Game
from tak.TakNNet import NNetWrapper as nn
from utils import *

args = dotdict({
    'numIters': 1000,
    'numEps': 100,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'arenaCompare': 40,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

if __name__ == "__main__":
    g = Game(5)  # 5x5 board
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    c.learn()
```

### Testing

Test the implementation:

```python
from tak.TakGame import TakGame

g = TakGame(5)
board = g.getInitBoard()
g.display(board)

valid_moves = g.getValidMoves(board, 1)
print(f"Number of valid moves: {sum(valid_moves)}")
```

## Action Encoding

Actions are encoded as integers:

### Placement Actions (0 to 3n²-1)
- Place flat/standing/capstone at each board position
- Formula: `row * n + col + piece_type * n²`
- piece_type: 0=flat, 1=standing, 2=capstone

### Movement Actions (3n² onwards)
- Pick up pieces and move in a direction with specific drop patterns
- Enumerated for all valid pickup counts and drop configurations

For a 5×5 board: ~3200 total actions

## Board Representation

The board is a 3D numpy array of shape `(height, n, n)` where:
- height = n (maximum stack size)
- Each cell contains values:
  - 0 = empty
  - 1 = white flat, 2 = black flat
  - 3 = white standing, 4 = black standing
  - 5 = white capstone, 6 = black capstone

## Neural Network Architecture

The default architecture uses:
- ResNet with 4 residual blocks
- 128 convolutional channels
- Separate policy and value heads
- Dropout for regularization

## Notes

- Start with smaller boards (3x3 or 4x4) for faster training
- Adjust `num_channels` and `num_res_blocks` in `TakNNet.py` for larger/smaller networks
- Training from scratch can take several hours to days depending on hardware

## Implementation Status

✅ **Complete:**
- Board representation and display
- Placement actions (flat, standing, capstone)
- Valid move generation for placements
- Road win detection (BFS algorithm)
- Flat count wins
- Canonical form transformation
- String representation for hashing
- PyTorch ResNet neural network

⚠️ **Partial:**
- Movement actions (basic structure, needs full validation)
- Opening move rules (not enforced in AlphaZero version)

📋 **TODO:**
- [ ] Complete movement action validation
- [ ] Piece count tracking in game state
- [ ] Opening move special rules in AlphaZero
- [ ] Symmetry augmentation for training
- [ ] Board rotation/reflection transformations
