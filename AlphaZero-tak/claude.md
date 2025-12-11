# AlphaZero Tak - Project Context

## Overview
This is an AlphaZero implementation for the board game Tak (5x5). The AI uses Monte Carlo Tree Search (MCTS) combined with a neural network to learn to play Tak through self-play.

## Tak Game Rules Implemented
1. **Piece limits**: Each player has 21 flats + 1 capstone for 5x5 board
2. **Opening rule**: First piece placed by each player must be opponent's color (flats only)
3. **Movement**: Pieces can move in any direction with multi-drop support
4. **Capstone mechanics**: Capstones can flatten standing stones when moving onto them
5. **Movement restrictions**: Can't move onto walls or capstones (except capstone flattening wall)
6. **Win condition**: Connect opposite edges with a road of flats/capstones (standing stones don't count)

## Key Files

### Core Game Logic
- **`tak/TakGame.py`**: Main game implementation
  - Board representation: `(n+4, n, n)` numpy array (game layers + piece count layers)
  - Action encoding: placements (0 to 3n²-1), movements (3n² onwards)
  - Movement patterns defined in `self.movement_patterns`
  - **IMPORTANT**: Piece limit check uses `>= 1.0` for flats, `>= 0.5` for capstones (floating-point precision)
  - Opening rule implemented in `getNextState()` (lines 166-182)
  - Valid moves generation includes movement validation via `_is_valid_movement()`

- **`tak/TakNNet.py`**: Neural network wrapper
  - ResNet architecture with PyTorch
  - **CRITICAL**: `board_channels, board_height, board_width = game.getBoardSize()` order matters!

### Training Components
- **`MCTS.py`**: Monte Carlo Tree Search with depth limiting (MAX_DEPTH = 200)
- **`Coach.py`**: Training loop with:
  - Move penalty system: Rewards shorter wins (lines 70-74)
  - Draw penalty: -0.2 instead of 0 to discourage draws (line 80)
  - Max moves: 600 per game (line 52)
- **`Arena.py`**: Model comparison arena (max_moves = 600, line 44)
- **`main_tak.py`**: Training configuration
  - Current: 100 iterations, 25 episodes/iter, 10 MCTS sims
  - Set to load existing model and continue training

### Interactive Play
- **`play_interactive.py`**: Human vs AI gameplay
  - Movement format: `m <from_row> <from_col> <direction> <pickup> [drops]`
  - Placement format: `<piece> <row> <col>` (f/s/c for flat/standing/capstone)
  - Example: `m 1 2 u 1 1` or `f 2 3`

### Tests
- **`tak/test_tak_rules.py`**: Comprehensive rule testing (6 tests)
- **`tak/test_valid_moves.py`**: Valid move generation tests
- **`tak/test_movement.py`**: Movement-specific tests
- **`test_win_detection.py`**: Road win detection tests

## Important Implementation Details

### Board Representation (TakGame.py)
```python
# Shape: (9, 5, 5) for 5x5 board
# Layers:
# 0: Current player's flats
# 1: Current player's standing stones
# 2: Current player's capstones
# 3: Opponent's flats
# 4: Opponent's standing stones
# 5: Opponent's capstones
# 6: Remaining flats (current player)
# 7: Remaining capstones (current player)
# 8: Opponent's remaining pieces (encoded)
```

### Action Encoding
- **Placements**: `action = row * n + col + piece_type * n * n`
  - piece_type: 0=flat, 1=standing, 2=capstone
- **Movements**: `action = num_placement_actions + position + direction * n² + pattern_idx * 4 * n²`
  - position: row * n + col
  - direction: 0=up, 1=down, 2=left, 3=right

### Critical Fixes Applied
1. **Floating-point precision**: Piece counts use `>= 1.0` instead of `> 0`
2. **Neural net dimensions**: Correct channel ordering for Conv2d
3. **MCTS depth limiting**: Prevents infinite recursion (MAX_DEPTH = 200)
4. **Move penalty system**: Incentivizes faster wins
5. **Valid move generation**: Includes movement action validation

## Training

### Quick Test (5 iterations)
```bash
# Edit main_tak.py: numIters=5, load_model=False
./venv/Scripts/python.exe main_tak.py
```

### Overnight Training (100 iterations)
```bash
# Edit main_tak.py: numIters=100, load_model=True
./venv/Scripts/python.exe main_tak.py
```

### Play Against Trained Model
```bash
./venv/Scripts/python.exe play_interactive.py
```

## Model Storage
- **Training checkpoint**: `temp/` (git-ignored)
- **Saved models**: `models/<name>/best.pth.tar`
- Current model: `models/5x5_easy_v2/` (5 iterations, 10 MCTS sims)

## Configuration Notes
- Board size: 5x5 (set in `main_tak.py`, line 36)
- MCTS simulations: 10 for easy, 25+ for stronger play
- Move cap: 600 (Coach.py:52, Arena.py:44)
- MCTS depth limit: 200 (MCTS.py:79) - **DO NOT change this**
- Temperature threshold: 15 moves (exploratory → deterministic)

## Known Behavior
- Early training produces many draws (expected)
- Move penalty system encourages aggressive play over time
- Opening rule means first 2 pieces are opponent's colors
- Capstones are rare (only 1 per player) - use strategically

## Git Workflow
```bash
git status                    # Check changes
git add .                     # Add all changes
git commit -m "message"       # Commit
git push                      # Push to remote
```

## Common Issues
1. **RecursionError**: Check MCTS.py has `depth` parameter in recursive call
2. **Dimension mismatch**: Verify TakNNet.py channel ordering
3. **Invalid moves**: Run `tak/test_valid_moves.py` to verify move generation
4. **No wins detected**: Run `test_win_detection.py` to verify road detection

## Next Steps for Improvement
- Train for more iterations (100+)
- Increase MCTS simulations for stronger play (25-50)
- Test against human players to validate strength
- Consider larger board sizes (6x6, 8x8) after 5x5 is solid
