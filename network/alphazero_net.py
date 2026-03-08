import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, num_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class AlphaZeroNet(nn.Module):
    def __init__(self, input_channels, board_shape, action_size,
                 num_res_blocks=2, num_filters=256):
        super().__init__()
        self.board_shape = board_shape
        self.action_size = action_size
        board_area = board_shape[0] * board_shape[1]

        # Initial conv block
        self.conv = nn.Conv2d(input_channels, num_filters, 3, padding=1)
        self.bn = nn.BatchNorm2d(num_filters)

        # Residual blocks
        self.res_blocks = nn.ModuleList(
            [ResBlock(num_filters) for _ in range(num_res_blocks)]
        )

        # Value head
        self.value_conv = nn.Conv2d(num_filters, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_area, 32)
        self.value_fc2 = nn.Linear(32, 1)

        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_area, action_size)

    def forward(self, x):
        # Backbone
        x = F.relu(self.bn(self.conv(x)))
        for block in self.res_blocks:
            x = block(x)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = F.softmax(self.policy_fc(p), dim=1)

        return v, p

    @torch.no_grad()
    def predict(self, state_input):
        """Run inference on a single state input.

        Args:
            state_input: numpy array of shape (C, H, W)

        Returns:
            (value, policy) where value is a float and policy is a numpy array
        """
        self.eval()
        x = torch.FloatTensor(state_input).unsqueeze(0)
        v, p = self(x)
        return v.item(), p.squeeze(0).numpy()

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(directory, f"{timestr}.pt")
        torch.save(self.state_dict(), path)

        latest_path = os.path.join(directory, "latest.txt")
        second_latest = ""
        if os.path.exists(latest_path):
            with open(latest_path) as f:
                second_latest = f.read().strip()
        with open(latest_path, "w") as f:
            f.write(f"{timestr}.pt")
        with open(os.path.join(directory, "second_latest.txt"), "w") as f:
            f.write(second_latest)

        return path

    def load_latest(self, directory):
        latest_path = os.path.join(directory, "latest.txt")
        if os.path.exists(latest_path):
            with open(latest_path) as f:
                name = f.read().strip()
            return self.load(os.path.join(directory, name))
        # Fall back to best.pt (e.g. fresh clone without latest.txt)
        best_path = os.path.join(directory, "best.pt")
        return self.load(best_path)

    def load(self, path):
        if not os.path.exists(path):
            return False
        self.load_state_dict(torch.load(path, weights_only=True))
        return True
