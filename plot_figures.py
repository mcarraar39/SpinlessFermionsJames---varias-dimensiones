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
parser.add_argument("--preepochs",          type=int,   default=5000, help="Number of pre-epochs for the pretraining phase")
parser.add_argument("--epochs",             type=int,   default=5000, help="Number of epochs for the energy minimisation phase")
parser.add_argument("-C", "--chunks",       type=int,   default=1,     help="Number of chunks for vectorized operations")

parser.add_argument("-B","--num_batches",   type=int,   default=10000, help="Number of batches of samples (effectively the length of the chain)")
parser.add_argument("-W","--num_walkers",   type=int,   default=10096,  help="Number of walkers used to generate configuration")
parser.add_argument("--num_sweeps",         type=int,   default=10,    help="Number of sweeped/discard proposed configurations between accepted batches (The equivalent of thinning constant)")
parser.add_argument("--dtype",              type=str,   default='float32',      help='Default dtype')
parser.add_argument("-Dim", "--dimensions", type=int,   default=1,     help="Number of dimensions of the physical system")
parser.add_argument("--comp", choices=['x','y','z'], default='x',
                    help="Component to show for natural-orbitals/OBDM if D > 1")
args = parser.parse_args()
chosen_comp = args.comp                   # 'x' por defecto

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
dimensions  = int(loaded.get('dimensions', args.dimensions))      # «1» si no existe

########################################################################################################################################################################
####################################################################### Density ########################################################################################
########################################################################################################################################################################
if dimensions == 1:
    density_xx = loaded['density_xx']
    density_psi = loaded['density_psi']

    plt.title('Density profile n(x) for %i fermions' % (nfermions))
    plt.xlabel('Position, x')
    plt.ylabel('Density, n(x)')
    #plt.ylim(0, 2)
    plt.plot(density_xx, density_psi)
    plt.show()

else:
    density_grid = loaded['density_grid'] #shape N^D
    density_edges = loaded['density_edges'] #list of edge arrays
    X, Y = density_edges[0], density_edges[1]  # Assuming 2D for simplicity
    proj_xy = density_grid.sum(axis=tuple(range(2, dimensions))) \
              if dimensions > 2 else density_grid
    plt.pcolormesh(X, Y, proj_xy, cmap='viridis')
    plt.gca().set_aspect('equal')
    plt.colorbar()
    plt.title(f'Density n(x,y) (projected) for {nfermions} fermions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # 1D marginal n(x) to compare with 1D case
    n_x = density_grid.sum(axis=tuple(range(1, dimensions))) \
        * np.diff(Y)[0]** (dimensions - 1)  # Integrate over y
    cx = X[:-1] + np.diff(X) / 2.  # Get bin centers
    plt.plot(cx, n_x, label='n(x)')
    plt.xlabel('Position, x')
    plt.title('Marginal Density n(x) for %i fermions' % (nfermions))
    plt.show()

# radial profile if D == 3
if dimensions == 3: 
    cx, cy, cz = [e[:-1] + np.diff(e) / 2. for e in density_edges]
    Xc, Yc, Zc = np.meshgrid(cx, cy, cz, indexing='ij')
    rr = np.sqrt(Xc**2 + Yc**2 + Zc**2).flatten()  # Radial distance
    dens_flat = density_grid.flatten()
    rmax = rr.max()
    nbins_r = 200
    n_r, r_edges = np.histogram(rr, bins=nbins_r, range=(0, rmax),
                                weights=dens_flat, density=True)
    rc = r_edges[:-1] + np.diff(r_edges) / 2.  # Bin centers
    plt.plot(rc, n_r)
    plt.xlabel('Radial distance, r')
    plt.ylabel('Density, n(r)')
    plt.title('Radial density profile n(r) for %i fermions' % (nfermions))
    plt.show()

########################################################################################################################################################################
####################################################################### OBDM ##############################################################################################
########################################################################################################################################################################

axes = 'xyz'[:dimensions] if dimensions >1 else 'x' # Axes for the density matrix

for ax in axes:
    if dimensions == 1:                                # --- caso unidimensional
        h_obdm      = loaded['h_obdm']
        xedges_obdm = loaded['xedges_obdm']
        yedges_obdm = loaded['yedges_obdm']
    else:                                              # --- D = 2, 3
        suf         = f'_{ax}'
        h_obdm      = loaded[f'h_obdm{suf}']
        xedges_obdm = loaded[f'xedges{suf}']
        yedges_obdm = loaded[f'yedges{suf}']

    cmap  = mpl.colormaps['seismic']
    norm  = colors.TwoSlopeNorm(vmin=h_obdm.min(),
                                vmax=h_obdm.max(), vcenter=0)
    print(f"The OBDM is {h_obdm}")
    plt.pcolormesh(xedges_obdm, yedges_obdm, h_obdm,
                   cmap=cmap, norm=norm)
    plt.colorbar()
    plt.xlabel(f'{ax}\u2081'); plt.ylabel(f"{ax}'\u2081")
    plt.title(f'OBDM $\\rho_{{{ax}{ax}}}$ for {nfermions} fermions')
    plt.show()

########################################################################################################################################################################
####################################################################### Natural orbitals ##########################################################################################
if dimensions == 1:
    eigenvalues = loaded['eigenvalues']
    eigenvectors = loaded['eigenvectors']
    xedges = loaded['xedges_obdm']
    label_ax = 'x'
else:
    lab = chosen_comp 
    try:
        eigenvalues = loaded[f'eigenvalues{("_" + lab) if lab else ""}']
        eigenvectors = loaded[f'eigenvectors{("_" + lab) if lab else ""}']
        xedges = loaded[f'xedges{("_"+lab) if lab else ""}']
        label_ax = lab
    except KeyError: # if 'z' is asked for and it is not available we use 'x'
        lab = 'x'
        eigenvalues = loaded['eigenvalues_x']
        eigenvectors = loaded['eigenvectors_x']
        xedges = loaded[f'xedges_x']
        label_ax = lab
#Occupation values and entropy
_eigen = eigenvalues[:nfermions]
entropy = -np.sum(_eigen * np.log(_eigen))
max_entropy = np.log(nfermions)
print("Occupation Values: ",_eigen)
print(f"Entropy: {entropy} | Max: {max_entropy}")

#grid of the OBDM

xvals  = xedges[:-1] + np.diff(xedges)/2

for i in range(nfermions):
    plt.plot(xvals, eigenvectors[:,i], label="%i" % (i))
plt.legend()
plt.title('Eigenstates of the one-body density matrix for %i fermions' % (nfermions))
plt.xlabel('Position, x')
plt.ylabel('Amplitude')
plt.show()

########################################################################################################################################################################
####################################################################### Pair correlation function ##############################################################################
##########################################################################################################################################################################
# if dimensions == 1:
#     h_tbd = loaded['h_tbd']
#     xedges_tbd = loaded['xedges_tbd']
#     yedges_tbd = loaded['yedges_tbd']

#     sc_tbdm=plt.pcolormesh(xedges_tbd, yedges_tbd, h_tbd, cmap=cc.cm.fire)
#     plt.colorbar(sc_tbdm)
#     plt.title('Density contour of the pair correlation function for %i fermions' % (nfermions))
#     plt.xlabel('Position, x_1')
#     plt.ylabel('Position, x_2')
#     plt.show()

#     # Calculate the x-values (positions) by getting bin centers
#     xvals = xedges_tbd[:-1] + np.diff(xedges_tbd)/2.

#     # Extract the diagonal elements of the two-body density matrix (h_tbd)
#     # This gives the probability of finding two particles at the same position
#     pair_correlation = np.diagonal(h_tbd, offset=0, axis1=-2, axis2=-1)

#     # Plot the pair correlation function
#     plt.plot(xvals, pair_correlation)
#     plt.title('Pair correlation function for %i fermions' % (nfermions))
#     plt.xlabel('Position, x')
#     plt.ylabel('Probability')
#     plt.show()
# else:
#     g2_r   = loaded['g2_r'];    r_edges = loaded['r_edges']
#     rc     = r_edges[:-1] + np.diff(r_edges)/2
#     plt.plot(rc, g2_r); plt.xlabel('r'); plt.ylabel('g₂(r)')
#     plt.title('Radial pair-correlation g₂(r)'); plt.show()