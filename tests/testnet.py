import torch
from torch import nn, Tensor
import numpy as np

import os, sys, time
# Add the parent directory (where "src/" is located) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


from Models import vLogHarmonicNet  # or whatever your class is called

# Parameters
nwalkers = 4096     # batch size
A = 6             # number of particles
D = 2             # dimensions per particle
num_hidden = 64
num_layers = 3
num_dets = 1
activation = torch.nn.Tanh()

# Instantiate the network
net = vLogHarmonicNet(
    num_input=A,
    num_hidden=num_hidden,
    num_layers=num_layers,
    num_dets=num_dets,
    func=activation,
    pretrain=False,
    Dim=D
)

# Create dummy input: shape [nwalkers, A, D]
x = torch.randn(nwalkers, A, D)
print(x.shape)
# Forward pass
sign, logabs = net(x)

# Print shapes
print("Input shape:     ", x.shape)
print("Output log|ψ| shape:", logabs.shape)   # Expected: [nwalkers]
print("Sign shape:      ", sign.shape)         # Expected: [nwalkers]