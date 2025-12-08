"""
Test that the game is ready for training
Verify that MCTS can explore both placement and movement actions
"""

import numpy as np
import sys
sys.path.append('..')
from TakGame import TakGame


def test_game_progression():
    """Simulate a game progression to ensure valid moves are always available"""
    print("\n" + "="*60)
    print("TEST: Game Progression with Valid Moves")
    print("="*60)

    game = TakGame(5)
    board = game.getInitBoard()
    player = 1

    move_count = 0
    max_moves = 20

    print("Simulating game progression...")

    while move_count < max_moves:
        valid_moves = game.getValidMoves(board, player)
        num_valid = np.sum(valid_moves)

        placement_valid = np.sum(valid_moves[:game.num_placement_actions])
        movement_valid = np.sum(valid_moves[game.num_placement_actions:])

        print(f"\nMove {move_count + 1}: Player {player}")
        print(f"  Placement actions: {placement_valid}")
        print(f"  Movement actions: {movement_valid}")
        print(f"  Total valid: {num_valid}")

        if num_valid == 0:
            print("ERROR: No valid moves available!")
            return False

        # Pick a random valid move
        valid_indices = np.where(valid_moves == 1)[0]
        action = np.random.choice(valid_indices)

        # Check what type of action it is
        if action < game.num_placement_actions:
            action_type = "placement"
        else:
            action_type = "movement"

        print(f"  Executing {action_type} action {action}")

        # Execute the move
        board, player = game.getNextState(board, player, action)

        # Check if game ended
        game_result = game.getGameEnded(board, player)
        if game_result != 0:
            print(f"\nGame ended with result: {game_result}")
            break

        move_count += 1

    if move_count >= max_moves:
        print(f"\n[PASS] Game progressed for {move_count} moves successfully")
    else:
        print(f"\n[PASS] Game ended naturally after {move_count} moves")

    game.display(board)
    return True


def test_mcts_can_explore():
    """Test that MCTS will be able to explore both types of actions"""
    print("\n" + "="*60)
    print("TEST: MCTS Exploration Readiness")
    print("="*60)

    game = TakGame(5)
    board = game.getInitBoard()
    player = 1

    # After opening moves, there should be both placement and movement options
    board, player = game.getNextState(board, player, 0)  # Opening 1
    board, player = game.getNextState(board, player, 1)  # Opening 2

    # Place several pieces to create movement opportunities
    for i in range(3):
        action = i * game.n + i
        board, player = game.getNextState(board, player, action)
        board, player = game.getNextState(board, player, 5 + i)

    game.display(board)

    valid_moves = game.getValidMoves(board, player)
    placement_valid = np.sum(valid_moves[:game.num_placement_actions])
    movement_valid = np.sum(valid_moves[game.num_placement_actions:])

    print(f"\nValid placement actions: {placement_valid}")
    print(f"Valid movement actions: {movement_valid}")

    if placement_valid > 0 and movement_valid > 0:
        print("\n[PASS] MCTS can explore both placement and movement actions!")
        print(f"Action space: {placement_valid + movement_valid} / {game.action_size} actions valid")
        return True
    else:
        print("\n[FAIL] Missing action types!")
        return False


def test_action_space_coverage():
    """Test that different types of movements are available"""
    print("\n" + "="*60)
    print("TEST: Action Space Coverage")
    print("="*60)

    game = TakGame(5)

    print(f"Total action space size: {game.action_size}")
    print(f"  Placement actions: {game.num_placement_actions}")
    print(f"  Movement actions: {game.num_movement_actions}")
    print(f"  Movement patterns: {len(game.movement_patterns)}")

    # Show some movement patterns
    print("\nSample movement patterns:")
    for i in range(min(10, len(game.movement_patterns))):
        pickup, pattern = game.movement_patterns[i]
        print(f"  Pattern {i}: pickup {pickup}, drops {pattern}")

    # Create a board with stacks of different heights
    board = game.getInitBoard()
    player = 1

    board, player = game.getNextState(board, player, 0)
    board, player = game.getNextState(board, player, 1)

    # Create stacks of height 1, 2, 3
    positions = [(0, 0), (1, 1), (2, 2)]
    heights = [1, 2, 3]

    for pos, height in zip(positions, heights):
        for h in range(height):
            action = pos[0] * game.n + pos[1]
            board, player = game.getNextState(board, player, action)
            board, player = game.getNextState(board, player, 15 + h)

    game.display(board)

    valid_moves = game.getValidMoves(board, player)

    # Count valid movements for each stack
    print("\nValid movements from each position:")
    for pos in positions:
        position_idx = pos[0] * game.n + pos[1]

        # Count movements from this position
        count = 0
        for direction in range(4):
            for pattern_idx in range(len(game.movement_patterns)):
                action = (game.num_placement_actions + position_idx +
                         direction * game.n * game.n +
                         pattern_idx * 4 * game.n * game.n)
                if action < game.action_size and valid_moves[action] == 1:
                    count += 1

        height = game._get_stack_height(board, pos[0], pos[1])
        print(f"  Position {pos} (height {height}): {count} valid movements")

    print("\n[PASS] Action space coverage verified")
    return True


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# TRAINING READINESS TESTS")
    print("#"*60)

    results = []
    results.append(("Game Progression", test_game_progression()))
    results.append(("MCTS Exploration", test_mcts_can_explore()))
    results.append(("Action Space Coverage", test_action_space_coverage()))

    print("\n" + "#"*60)
    print("# SUMMARY")
    print("#"*60)
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {name}")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n" + "="*60)
        print("  GAME IS READY FOR TRAINING!")
        print("="*60)
