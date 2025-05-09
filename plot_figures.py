'''This code plots the figures given '''
import torch
from torch import nn
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.colors as colors

import colorcet as cc

import argparse

parser = argparse.ArgumentParser(prog="SpinlessFermions",
                                 usage='%(prog)s [options]',
                                 description="A Neural Quantum State (NQS) solution to one-dimensional fermions interacting in a Harmonic trap",
                                 epilog="and fin")

parser.add_argument("-N", "--num_fermions", type=int,   default=2,     help="Number of fermions in physical system")
parser.add_argument("-H", "--num_hidden",   type=int,   default=64,    help="Number of hidden neurons per layer")
parser.add_argument("-L", "--num_layers",   type=int,   default=2,     help="Number of layers within the network")
parser.add_argument("-D", "--num_dets",     type=int,   default=1,     help="Number of determinants within the network's final layer")
parser.add_argument("-V", "--V0",           type=float, default=0.,    help="Interaction strength (in harmonic units)")
parser.add_argument("-S", "--sigma0",       type=float, default=0.5,   help="Interaction distance (in harmonic units")
parser.add_argument("--preepochs",          type=int,   default=10000, help="Number of pre-epochs for the pretraining phase")
parser.add_argument("--epochs",             type=int,   default=10000, help="Number of epochs for the energy minimisation phase")
parser.add_argument("-C", "--chunks",       type=int,   default=1,     help="Number of chunks for vectorized operations")

parser.add_argument("-B","--num_batches",   type=int,   default=10000, help="Number of batches of samples (effectively the length of the chain)")
parser.add_argument("-W","--num_walkers",   type=int,   default=4096,  help="Number of walkers used to generate configuration")
parser.add_argument("--num_sweeps",         type=int,   default=10,    help="Number of sweeped/discard proposed configurations between accepted batches (The equivalent of thinning constant)")
parser.add_argument("--dtype",              type=str,   default='float32',      help='Default dtype')
parser.add_argument("-Dim", "--dimensions", type=int,   default=1,     help="Number of dimensions of the physical system")

args = parser.parse_args()

#set the default dtype
if(args.dtype == 'float32'):
    torch.set_default_dtype(torch.float32)
    dtype = 'float32'
elif(args.dtype == 'float64'):
    torch.set_default_dtype(torch.float64)
    dtype = 'float64'
else:
    raise NameError(f"Unknown dtype: {args.dtype} selected!")

nfermions = args.num_fermions #number of input nodes
num_hidden = args.num_hidden  #number of hidden nodes per layer
num_layers = args.num_layers  #number of layers in network
num_dets = args.num_dets      #number of determinants (accepts arb. value)
func = nn.Tanh()  #activation function between layers
pretrain = True   #pretraining output shape?

#nbatches = args.num_batches
nwalkers=args.num_walkers
n_sweeps=args.num_sweeps #n_discard
std=1.#0.02#1.
target_acceptance=0.5

V0 = args.V0
sigma0 = args.sigma0

pt_save_every_ith=1000
em_save_every_ith=1000

nchunks=1

preepochs=args.preepochs
epochs=args.epochs
dimensions=args.dimensions

optim = "Adam"

device='cpu'#'cuda'
#dtype='float32'

analysis_datapath = "analysis/PHYS_A%02i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_V%4.2e_S%4.2e_%s_PT_%s_device_%s_dtype_%s_dim_%02i.npz" % \
                (nfermions, num_hidden, num_layers, num_dets, func.__class__.__name__, nwalkers, preepochs, V0, sigma0, \
                 optim, False, device, dtype,dimensions)#load the data to plot

loaded = np.load(analysis_datapath)

density_xx = loaded['density_xx']
density_psi = loaded['density_psi']

plt.title('Density profile n(x) for %i fermions' % (nfermions))
plt.xlabel('Position, x')
plt.ylabel('Density, n(x)')
#plt.ylim(0, 2)
plt.plot(density_xx, density_psi)
plt.show()

h_obdm = loaded['h_obdm']           #rho-matrix (OBDM)
xedges_obdm = loaded['xedges_obdm'] 
yedges_obdm = loaded['yedges_obdm']

cmap=mpl.colormaps['seismic'] #cwr #plt.cm.bwr
norm=colors.TwoSlopeNorm(vmin=np.min(h_obdm), vmax=np.max(h_obdm), vcenter=0.)

sc=plt.pcolormesh(xedges_obdm, yedges_obdm, h_obdm, cmap=cmap, norm=norm)
#plt.contour(xedges, yedges, rho_matrix, color='black')
plt.title('Density contour of the one-body density matrix for %i fermions' % (nfermions))
plt.xlabel('Position, x_1')
plt.ylabel('Position, x`_1')
plt.colorbar(sc)
plt.show()

eigenvalues = loaded['eigenvalues']
eigenvectors = loaded['eigenvectors']

_eigen = eigenvalues[:nfermions]
entropy = -np.sum(_eigen * np.log(_eigen))
max_entropy = np.log(nfermions)

print("Occupation Values: ",_eigen)
print(f"Entropy: {entropy} | Max: {max_entropy}")

xvals = xedges_obdm[:-1] + np.diff(xedges_obdm)/2.

for i in range(nfermions):
    plt.plot(xvals, eigenvectors[:,i], label="%i" % (i))
plt.legend()
plt.title('Eigenstates of the one-body density matrix for %i fermions' % (nfermions))
plt.xlabel('Position, x')
plt.ylabel('Amplitude')
plt.show()

h_tbd = loaded['h_tbd']
xedges_tbd = loaded['xedges_tbd']
yedges_tbd = loaded['yedges_tbd']

sc_tbdm=plt.pcolormesh(xedges_tbd, yedges_tbd, h_tbd, cmap=cc.cm.fire)
plt.colorbar(sc_tbdm)
plt.title('Density contour of the pair correlation function for %i fermions' % (nfermions))
plt.xlabel('Position, x_1')
plt.ylabel('Position, x_2')
plt.show()

# Calculate the x-values (positions) by getting bin centers
xvals = xedges_tbd[:-1] + np.diff(xedges_tbd)/2.

# Extract the diagonal elements of the two-body density matrix (h_tbd)
# This gives the probability of finding two particles at the same position
pair_correlation = np.diagonal(h_tbd, offset=0, axis1=-2, axis2=-1)

# Plot the pair correlation function
plt.plot(xvals, pair_correlation)
plt.title('Pair correlation function for %i fermions' % (nfermions))
plt.xlabel('Position, x')
plt.ylabel('Probability')
plt.show()
