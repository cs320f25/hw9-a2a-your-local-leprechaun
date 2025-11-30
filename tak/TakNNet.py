"""
TakNNet.py - Neural Network Wrapper for Tak
Handles training, prediction, and checkpoint management
"""

import os
import sys
import time
import numpy as np
import torch
import torch.optim as optim

from pytorch.NNet import TakNNet as TakNNetModel
from pytorch.NNet import dotdict

sys.path.append('..')

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 128,
    'num_res_blocks': 4,
})


class NNetWrapper:
    """
    Wrapper class for the PyTorch neural network.
    Provides interface for alpha-zero-general framework.
    """

    def __init__(self, game):
        self.game = game
        self.board_x, self.board_y, self.board_z = game.getBoardSize()
        self.action_size = game.getActionSize()

        # Initialize neural network
        self.nnet = TakNNetModel(game, args)

        if args.cuda:
            self.nnet.cuda()

        # Optimizer
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr)

    def train(self, examples):
        """
        Train the neural network on examples.

        Args:
            examples: List of (board, pi, v) tuples
                board: Board state
                pi: MCTS policy (action probabilities)
                v: Value target (-1, 0, 1)
        """
        self.nnet.train()

        for epoch in range(args.epochs):
            print(f'Epoch {epoch + 1}/{args.epochs}')

            batch_count = len(examples) // args.batch_size

            t = time.time()
            pi_losses = []
            v_losses = []

            for batch_idx in range(batch_count):
                # Sample batch
                sample_ids = np.random.choice(len(examples), size=args.batch_size, replace=False)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))

                # Convert to tensors
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # Move to GPU if available
                if args.cuda:
                    boards = boards.cuda()
                    target_pis = target_pis.cuda()
                    target_vs = target_vs.cuda()

                # Forward pass
                out_pi, out_v = self.nnet(boards)

                # Compute losses
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # Record losses
                pi_losses.append(l_pi.item())
                v_losses.append(l_v.item())

                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

            print(f'Time: {time.time() - t:.2f}s')
            print(f'Policy Loss: {np.mean(pi_losses):.4f}, Value Loss: {np.mean(v_losses):.4f}')

    def predict(self, board):
        """
        Predict policy and value for a single board state.

        Args:
            board: Board state numpy array

        Returns:
            pi: Action probabilities (numpy array)
            v: Position value (float)
        """
        self.nnet.eval()

        # Prepare input
        board = torch.FloatTensor(board.astype(np.float64))
        if args.cuda:
            board = board.cuda()
        board = board.view(1, self.board_z, self.board_x, self.board_y)

        # Forward pass
        with torch.no_grad():
            pi, v = self.nnet(board)

        # Convert to numpy
        pi = torch.exp(pi).cpu().numpy()[0]
        v = v.cpu().numpy()[0][0]

        return pi, v

    def loss_pi(self, targets, outputs):
        """Policy loss: cross-entropy between target and predicted policies."""
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        """Value loss: mean squared error."""
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        """Save model checkpoint."""
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(f"Checkpoint Directory does not exist! Making directory {folder}")
            os.mkdir(folder)

        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        """Load model checkpoint."""
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model in path {filepath}")

        checkpoint = torch.load(filepath, map_location='cpu' if not args.cuda else None)
        self.nnet.load_state_dict(checkpoint['state_dict'])
