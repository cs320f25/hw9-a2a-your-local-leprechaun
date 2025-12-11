# AlphaZero-Tak: AI Engineering Final Project

An implementation of the strategic board game Tak using the AlphaZero reinforcement learning framework, complete with intelligent agents built using Google's Agent Development Kit (ADK).

## Table of Contents
- [About Tak](#about-tak)
- [Project Overview](#project-overview)
- [Game Rules](#game-rules)
- [Getting Started](#getting-started)
- [Project Components](#project-components)
- [Play Against the AI](#play-against-the-ai)
- [The Agents](#the-agents)
- [Current Status](#current-status)
- [Limitations](#limitations)
- [Future Work](#future-work)

## About Tak

Tak is a beautiful abstract strategy game introduced in Patrick Rothfuss's *Kingkiller Chronicles* series. The game combines elements of Go, Chess, and Backgammon into an elegant system where players vie to create a road connecting opposite edges of the board.

## Project Overview

This project consists of three main components:

1. **AlphaZero-Tak**: A neural network-based AI trained using the AlphaZero algorithm to play Tak
2. **Takbot**: A conversational agent that answers questions about Tak rules, strategies, and mechanics
3. **Takbot-Status**: A status monitoring agent for tracking training progress and system information

The project began by implementing the complete Tak game engine in Python with proper rule enforcement, then porting it to the AlphaZero framework for neural network training via self-play.

## Game Rules

### Basic Objective
**Win by creating a road** - Connect two opposite edges of the board with a continuous line of your pieces (flats or capstones).

### Pieces
Players have three types of pieces:
- **Flats**: Standard pieces that count toward roads and stacks
- **Standing Stones (Walls)**: Block roads but don't count toward building them
- **Capstones**: Powerful pieces that count toward roads and can flatten standing stones

Piece counts vary by board size (5x5 board has 21 flats and 1 capstone per player).

### Gameplay
1. **Opening Moves**: Each player's first move places an opponent's flat stone
2. **Regular Play**: On your turn, either:
   - **Place** a piece (flat, standing stone, or capstone) on an empty square
   - **Move** a stack you control by picking up pieces and dropping them along a line

### Movement Rules
- You control a stack if your piece is on top
- Pick up some or all pieces from your stack (up to board size)
- Move in a straight line (up/down/left/right)
- Drop at least one piece per square
- Capstones can flatten standing stones when moving onto them

### Winning Conditions
1. **Road Win**: First player to connect opposite edges wins immediately
2. **Flat Win**: If the board fills up, player with the most flat stones on top wins
3. **Draw**: Equal number of flats at game end

## Getting Started

### Prerequisites
- Python 3.7+
- PyTorch
- NumPy
- (Optional) Google Cloud SDK for agent deployment

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd hw9-a2a-your-local-leprechaun
```

2. Install AlphaZero dependencies:
```bash
cd AlphaZero-tak
pip install -r requirements.txt
```

3. (Optional) Set up the agents:
```bash
cd ../takbot
make install
# or
cd ../takbot-status
make install
```

## Project Components

### AlphaZero-Tak (`/AlphaZero-tak/`)
The core game engine and AI training system:
- `tak/TakGame.py` - Game logic and rules implementation
- `tak/TakLogic.py` - Low-level board operations and win detection
- `tak/TakNNet.py` - Neural network architecture
- `Coach.py` - Training orchestration via self-play
- `MCTS.py` - Monte Carlo Tree Search for move selection
- `play.py` - Interactive human vs AI gameplay script

### Takbot (`/takbot/`)
A conversational AI agent that serves as a Tak rules expert:
- Answers questions about game rules and strategies
- Provides move suggestions and explanations
- Built with Google's Agent Development Kit (ADK)
- Deployed via Google Cloud Platform

### Takbot-Status (`/takbot-status/`)
A monitoring agent for the training system:
- Tracks model training progress
- Reports system metrics
- Provides status updates on training iterations
- Built with Google's Agent Development Kit (ADK)

## Play Against the AI

To play an interactive game against the trained neural network:

### Quick Start
```bash
cd AlphaZero-tak
python play.py
```

### Configuration
Edit the configuration section in `play.py` to customize your experience:

```python
# Model settings
MODEL_PATH = './models/5x5_easy_v3/'  # Path to trained model
MODEL_FILE = 'best.pth.tar'           # Model checkpoint

# Game settings
BOARD_SIZE = 5          # Board size (must match trained model)
AI_STRENGTH = 25        # MCTS simulations (10=easy, 25=medium, 50+=hard)
HUMAN_GOES_FIRST = True # Set False to let AI play first
```

### How to Play

When it's your turn, enter moves using these formats:

**Place a piece:**
```
<piece_type> <row> <col>

Examples:
  f 2 3  - Place a flat stone at row 2, column 3
  s 1 1  - Place a standing stone at row 1, column 1
  c 0 0  - Place a capstone at row 0, column 0
```

**Move a stack:**
```
m <from_row> <from_col> <direction> <pickup> [drops...]

Examples:
  m 1 2 u 1 1      - Move 1 piece from (1,2) up, drop it
  m 2 2 r 3 2 1    - Move 3 pieces from (2,2) right, drop 2 then 1
```

**Directions:**
- `u` or `up` - Move up (north)
- `d` or `down` - Move down (south)
- `l` or `left` - Move left (west)
- `r` or `right` - Move right (east)

**Other commands:**
- `quit` or `q` - Exit the game

### Tips
- Coordinates are 0-indexed (0 to 4 for a 5x5 board)
- Only flats and capstones count toward building roads
- Standing stones block roads but can be strategic for defense
- The AI displays its moves and reasoning as it plays

## The Agents

### Takbot
A specialized Tak assistant that helps players learn the game:
- Explains rules and edge cases
- Provides strategic advice
- Analyzes board positions
- Available via local playground or deployed to GCP

**Local Testing:**
```bash
cd takbot
make install && make playground
```

**Deployment:**
```bash
cd takbot
make deploy
```

### Takbot-Status
A monitoring companion for the AlphaZero training process:
- Reports on training iteration progress
- Tracks model performance metrics
- Provides system health updates
- Monitors resource usage during training

**Local Testing:**
```bash
cd takbot-status
make install && make playground
```

**Deployment:**
```bash
cd takbot-status
make deploy
```

Both agents are built using Google's Agent Development Kit and follow a "bring your own agent" architecture, making them easily customizable and extensible.

## Current Status

### Completed
- Full Tak game implementation with proper rules and piece tracking
- AlphaZero framework integration
- Interactive play interface for human vs AI matches
- Basic neural network architecture for policy and value prediction
- 17 training iterations completed (model: `5x5_easy_v3`)
- Both agents (Takbot and Takbot-Status) successfully deployed to Google Cloud

### In Progress
- Refining the AlphaZero implementation to properly handle piece constraints
- Debugging edge cases in movement validation
- Improving neural network training efficiency

## Limitations

### Training Challenges
1. **Computational Cost**: Training AlphaZero requires significant computational resources and time. Each iteration involves thousands of self-play games followed by neural network updates.

2. **Convergence Time**: The current model (17 iterations) shows basic play but hasn't reached strong strategic understanding. Achieving master-level play typically requires hundreds of iterations.

3. **Piece Tracking**: Early versions had issues with piece count enforcement during self-play, requiring manual validation and debugging of the action space.

4. **Model Verification**: Testing AlphaZero implementations is challenging since the model only reveals issues during actual gameplay rather than during training.

### Technical Limitations
1. **Board Size**: Currently optimized for 5x5 boards. Larger boards (6x6, 8x8) would require architectural changes and significantly more training.

2. **Action Space Complexity**: Tak's movement patterns create a large action space (~3000+ possible actions), making the learning problem more difficult than games like Chess or Go.

3. **Evaluation**: Unlike Chess where engines exist for comparison, Tak has fewer benchmarks for measuring AI strength.

4. **Hardware Requirements**: Training requires a CUDA-capable GPU for reasonable performance. CPU-only training is prohibitively slow.


## Contributing

This project was created as a final project for AI Engineering Class. Contributions, suggestions, and feedback are welcome.

## Acknowledgments

- Based on the [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) framework
- Tak game design by James Ernest
- Inspired by the *Kingkiller Chronicles* by Patrick Rothfuss
- Built using Google's [Agent Development Kit (ADK)](https://github.com/GoogleCloudPlatform/agent-starter-pack)

## License

This project is for educational purposes.
