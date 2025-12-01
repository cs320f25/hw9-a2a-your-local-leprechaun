"""
NNet.py - PyTorch Neural Network for Tak
Implements a ResNet architecture for policy and value prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TakNNet(nn.Module):
    """
    ResNet-style neural network for Tak.

    Architecture:
    - Convolutional input layer
    - Multiple residual blocks
    - Policy head (outputs move probabilities)
    - Value head (outputs position evaluation)
    """

    def __init__(self, game, args):
        super(TakNNet, self).__init__()

        # getBoardSize() returns (depth/channels, width, height)
        self.board_z, self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Network parameters
        self.num_channels = args.num_channels if hasattr(args, 'num_channels') else 128
        self.num_res_blocks = args.num_res_blocks if hasattr(args, 'num_res_blocks') else 4
        self.dropout = args.dropout if hasattr(args, 'dropout') else 0.3

        # Initial convolutional layer
        # Input: (batch, board_z, board_x, board_y)
        self.conv_input = nn.Conv2d(
            in_channels=self.board_z,
            out_channels=self.num_channels,
            kernel_size=3,
            padding=1
        )
        self.bn_input = nn.BatchNorm2d(self.num_channels)

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(self.num_channels) for _ in range(self.num_res_blocks)
        ])

        # Policy head
        self.conv_policy = nn.Conv2d(self.num_channels, 32, kernel_size=1)
        self.bn_policy = nn.BatchNorm2d(32)
        self.fc_policy = nn.Linear(32 * self.board_x * self.board_y, self.action_size)

        # Value head
        self.conv_value = nn.Conv2d(self.num_channels, 3, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(3)
        self.fc_value1 = nn.Linear(3 * self.board_x * self.board_y, 64)
        self.fc_value2 = nn.Linear(64, 1)

        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, board_z, board_x, board_y)

        Returns:
            policy: Log probabilities over actions (batch, action_size)
            value: Position evaluation (batch, 1)
        """
        # Initial convolution
        x = F.relu(self.bn_input(self.conv_input(x)))

        # Residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Policy head
        p = F.relu(self.bn_policy(self.conv_policy(x)))
        p = p.view(p.size(0), -1)  # Flatten
        p = self.dropout_layer(p)
        p = self.fc_policy(p)
        policy = F.log_softmax(p, dim=1)

        # Value head
        v = F.relu(self.bn_value(self.conv_value(x)))
        v = v.view(v.size(0), -1)  # Flatten
        v = self.dropout_layer(v)
        v = F.relu(self.fc_value1(v))
        v = self.dropout_layer(v)
        v = torch.tanh(self.fc_value2(v))
        value = v

        return policy, value


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutional layers and skip connection.
    """

    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)

        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += residual  # Skip connection
        out = F.relu(out)

        return out


class dotdict(dict):
    """Helper class to access dict items as attributes."""
    def __getattr__(self, name):
        return self[name]
