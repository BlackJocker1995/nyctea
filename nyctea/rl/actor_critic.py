"""DDPG actor/critic networks (PyTorch).

Topology preserved verbatim from the legacy ``Rl/actor_critic.py``: three linear
layers each with ``normal_(0, 0.1)`` init; Actor uses tanh→sigmoid→sigmoid,
Critic concatenates state+action then tanh→sigmoid→linear.

The only change is dropping the unused ``gym`` / ``matplotlib`` imports and the
dead commented ``Buffer`` class.
"""
import torch
import torch.nn as nn


class Actor(nn.Module):
    """Policy network: state → action in [0, 1]^d (sigmoid output)."""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear1.weight.data.normal_(0, 0.1)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear2.weight.data.normal_(0, 0.1)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.linear3.weight.data.normal_(0, 0.1)

    def forward(self, s):
        x = torch.tanh(self.linear1(s))
        x = torch.sigmoid(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        return x


class Critic(nn.Module):
    """Action-value network: (state, action) → Q."""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear1.weight.data.normal_(0, 0.1)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear2.weight.data.normal_(0, 0.1)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.linear3.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        x = torch.cat([s, a], 1)
        x = torch.tanh(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = self.linear3(x)
        return x
