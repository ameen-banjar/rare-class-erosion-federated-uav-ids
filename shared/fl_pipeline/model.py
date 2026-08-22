import copy

import torch
from torch import nn


class AttackFamilyMLP(nn.Module):
    """10-way attack-family classifier (ISOT is single-label per session/row,
    not multi-label -- see README's "concurrent attack recall" correction)."""

    def __init__(self, input_dim, n_classes=10, hidden=(128, 64), dropout=0.2):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def get_state_dict_copy(model):
    return copy.deepcopy(model.state_dict())


def set_state_dict(model, state_dict):
    model.load_state_dict(state_dict)


def save_checkpoint(model, path):
    """Saves a model checkpoint so a later run can reuse it (e.g. to compute
    an additional metric) without a costly retrain."""
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, map_location=None):
    model.load_state_dict(torch.load(path, map_location=map_location))
    return model
