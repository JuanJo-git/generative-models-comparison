"""MLP Classifier 784 -> 256 -> 64 -> 10 (Bagian B).

z_MLP = aktivasi ReLU lapisan 64 dimensi (fitur penultimate).
Bias sengaja tetap ada (standar), tetapi proyeksi balik memakai transpos bobot saja.
"""
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def features(self, x):
        """z_MLP (B, 64), setelah ReLU."""
        h1 = F.relu(self.fc1(x.flatten(1)))
        return F.relu(self.fc2(h1))

    def forward(self, x):
        return self.fc3(self.features(x))

    def weights(self):
        """W1 (256x784), W2 (64x256), W3 (10x64) — konvensi nn.Linear: out x in."""
        return (self.fc1.weight.detach(), self.fc2.weight.detach(), self.fc3.weight.detach())
