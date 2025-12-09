"""
Training script for Tak using AlphaZero
"""

import logging
import coloredlogs

from Coach import Coach
from tak.TakGame import TakGame as Game
from tak.TakNNet import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

# Training parameters
args = dotdict({
    'numIters': 100,                # Number of training iterations (overnight training)
    'numEps': 25,                   # Number of games per iteration (episodes)
    'tempThreshold': 20,            # Moves before switching to deterministic play (increased exploration)
    'updateThreshold': 0.51,        # Threshold to accept new model (lower = accept improvements easier)
    'maxlenOfQueue': 20000,         # Max training examples to keep
    'numMCTSSims': 25,              # Number of MCTS simulations per move (increased for better moves)
    'arenaCompare': 20,             # Number of games to play when comparing models
    'cpuct': 1,                     # MCTS exploration constant

    'checkpoint': './temp/',        # Checkpoint directory
    'load_model': True,             # Load existing model (continue training)
    'load_folder_file': ('./temp/','best.pth.tar'),  # Model to load
    'numItersForTrainExamplesHistory': 20,  # Training history to keep
})

def main():
    log.info('Loading Tak Game...')
    g = Game(5)  # Start with 3x3 for faster training!

    log.info('Loading Neural Network...')
    nnet = nn(g)

    if args.load_model:
        log.info(f'Loading checkpoint "{args.load_folder_file[1]}" from {args.load_folder_file[0]}...')
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🚀')
    c.learn()

if __name__ == "__main__":
    main()
