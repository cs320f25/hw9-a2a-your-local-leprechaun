# 5x5 Easy Agent v1

## Training Configuration

**Board Size**: 5x5

**Training Parameters**:
- Iterations: 20
- Episodes per iteration: 25
- MCTS simulations: 10
- Update threshold: 0.55

**Network Architecture**:
- Channels: 64
- Residual blocks: 2
- Epochs: 5
- Learning rate: 0.001

## Usage

Load this model in your scripts:
```python
from tak.TakNNet import NNetWrapper as nn
from tak.TakGame import TakGame

game = TakGame(5)
nnet = nn(game)
nnet.load_checkpoint('./models/5x5_easy_v1/', 'best.pth.tar')
```
