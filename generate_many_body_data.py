'''This code generates the plotting data that will be used to generate the figures in the plot_figures.py script'''

import torch
from torch import nn, Tensor
import os, sys, time

torch.manual_seed(0)
torch.set_printoptions(4)
torch.backends.cudnn.benchmark=True
torch.set_default_dtype(torch.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #set device)
dtype = str(torch.get_default_dtype()).split('.')[-1]

sys.path.append("./src/")

from Models import vLogHarmonicNet
from Samplers import MetropolisHastings
from Hamiltonian import HarmonicOscillatorWithInteractionD as HOw1D
from Pretraining import HermitePolynomialMatrixND  

from utils import load_dataframe, load_model, count_parameters, get_groundstate
from utils import get_params, sync_time, clip, calc_pretraining_loss

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import colorcet as cc
import math

##################################################################################################################################################
#####                                  Setting the parser to configurate how the code is run from the terminal                               #####
##################################################################################################################################################
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

parser.add_argument("--mode", default='standard', choices=['standard','notrap','nobackflow'], help='Flag to select which system the standard (run.py), notrap (run_notrap.py), and nobackflow (run_nobackflow.py)')
parser.add_argument("-Dim", "--dimensions", type=int,   default=1,     help="Number of dimensions of the physical system")

args = parser.parse_args()

###############################################################################################################################################
#####                                           Setting the NQS model and the Hamiltonian                                                 #####
###############################################################################################################################################
nfermions = args.num_fermions #number of input nodes
num_hidden = args.num_hidden  #number of hidden nodes per layer
num_layers = args.num_layers  #number of layers in network
num_dets = args.num_dets      #number of determinants (accepts arb. value)
func = nn.Tanh()  #activation function between layers
pretrain = True   #pretraining output shape?

nbatches = args.num_batches
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

#torch.manual_seed(42)
net = vLogHarmonicNet(num_input=nfermions,
                      num_hidden=num_hidden,
                      num_layers=num_layers,
                      num_dets=num_dets,
                      func=func,
                      pretrain=pretrain,
                      Dim=dimensions)
net=net.to(device)

#torch.manual_seed(42)
sampler = MetropolisHastings(network=net,
                             dof=nfermions,
                             nwalkers=nwalkers,
                             target_acceptance=target_acceptance,
                             dim=dimensions)

###############################################################################################################################################
#####                                             GENERATE MANY-BODY DATA                                                                 #####
###############################################################################################################################################

net.pretrain = False #check it's false
optim = torch.optim.Adam(params=net.parameters(), lr=1e-4) #new optimizer

mode = args.mode
if(mode=='standard'):
    model_path = "results/energy/checkpoints/A%02i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_V%4.2e_S%4.2e_%s_PT_%s_device_%s_dtype_%s_dim_%02i_chkp.pt" % \
                    (nfermions, num_hidden, num_layers, num_dets, func.__class__.__name__, nwalkers, preepochs, V0, sigma0, \
                     optim.__class__.__name__, False, device, dtype,dimensions)
    filename = "results/energy/data/A%02i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_V%4.2e_S%4.2e_%s_PT_%s_device_%s_dtype_%s_dim_%02i.csv" % \
                    (nfermions, num_hidden, num_layers, num_dets, func.__class__.__name__, nwalkers, preepochs, V0, sigma0, \
                     optim.__class__.__name__, False, device, dtype,dimensions)
    analysis_datapath = "analysis/PHYS_A%02i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_V%4.2e_S%4.2e_%s_PT_%s_device_%s_dtype_%s_dim_%02i.npz" % \
                (nfermions, num_hidden, num_layers, num_dets, func.__class__.__name__, nwalkers, preepochs, V0, sigma0, \
                 optim.__class__.__name__, False, device, dtype,dimensions)
elif(mode=='notrap'):
    model_path = "results/energy/checkpoints/A%02i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_V%4.2e_S%4.2e_%s_PT_%s_NO_TRAP_device_%s_dtype_%s_dim_%02i_chkp.pt" % \
                    (nfermions, num_hidden, num_layers, num_dets, func.__class__.__name__, nwalkers, preepochs, V0, sigma0, \
                     optim.__class__.__name__, False, device, dtype,dimensions)
    filename = "results/energy/data/A%02i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_V%4.2e_S%4.2e_%s_PT_%s_NO_TRAP_device_%s_dtype_%s_dim_%02i.csv" % \
                    (nfermions, num_hidden, num_layers, num_dets, func.__class__.__name__, nwalkers, preepochs, V0, sigma0, \
                     optim.__class__.__name__, False, device, dtype,dimensions)
    analysis_datapath = "analysis/PHYS_notrap_A%02i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_V%4.2e_S%4.2e_%s_PT_%s_device_%s_dtype_%s_dim_%02i.npz" % \
                (nfermions, num_hidden, num_layers, num_dets, func.__class__.__name__, nwalkers, preepochs, V0, sigma0, \
                 optim.__class__.__name__, False, device, dtype,dimensions)
elif(mode=='nobackflow'):
    model_path = "results/energy/checkpoints/no_backflow_A%02i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_V%4.2e_S%4.2e_%s_PT_%s_device_%s_dtype_%s_dim_%02i_chkp.pt" % \
                    (nfermions, num_hidden, num_layers, num_dets, func.__class__.__name__, nwalkers, preepochs, V0, sigma0, \
                     optim.__class__.__name__, False, device, dtype,dimensions)
    filename = "results/energy/data/no_backflow_A%02i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_V%4.2e_S%4.2e_%s_PT_%s_device_%s_dtype_%s_dim_%02i.csv" % \
                    (nfermions, num_hidden, num_layers, num_dets, func.__class__.__name__, nwalkers, preepochs, V0, sigma0, \
                     optim.__class__.__name__, False, device, dtype,dimensions)
    analysis_datapath = "analysis/PHYS_no_backflow_A%02i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_V%4.2e_S%4.2e_%s_PT_%s_device_%s_dtype_%s_dim_%02i.npz" % \
                (nfermions, num_hidden, num_layers, num_dets, func.__class__.__name__, nwalkers, preepochs, V0, sigma0, \
                 optim.__class__.__name__, False, device, dtype,dimensions)
else:
    raise NameError(f"Unknown mode: {mode} selected!")

#Load model and create storage tensors
writer = load_dataframe(filename)
output_dict = load_model(model_path=model_path, device=device, net=net, optim=optim, sampler=sampler)

# Initialize storage tensors
configurations = torch.zeros(size=(nbatches, nwalkers, nfermions,dimensions), dtype=torch.get_default_dtype(), device='cpu') #store on CPU #modified 

xmin=-6
xmax=+6

xdata = torch.zeros([nbatches, nwalkers])
ydata = torch.zeros([nbatches, nwalkers])
zdata = torch.zeros([nbatches, nwalkers])

with torch.no_grad():
    for batch in tqdm(range(nbatches), desc="Generating configurations"):
        # Generate configurations using MCMC
        x, _ = sampler(n_sweeps=n_sweeps)

        # Generate random positions for ghost particle
        s = torch.rand_like(x)*(xmax-xmin) + xmin #uniform rand in range [xmin, xmax)

        # Store configurations
        configurations[batch, :, :,:] = x.detach().clone() #check clone

        # Ghost particle method for OBDM
        xp = x.clone() #ghost-particle method for computing one-body density matrix
        xp[:, 0, 0] = s[:, 0, 0] # Move first particle #modified

        # Calculate wavefunctions
        sgn_p, logabs_p = net(xp) # Ghost configuration
        sgn, logabs = net(x)      # Original configuration

        #print(x.shape, xp.shape)

        # Store data for OBDM calculation
        xdata[batch, :] = x[:, 0, 0]  # first particle, first dim #modified
        ydata[batch, :] = xp[:, 0, 0] # ghost position#modified
        zdata[batch, :] = ((xmax-xmin) * sgn_p*sgn * torch.exp(logabs_p-logabs)) #wavefunction ratio (the "matrix element")

def to_numpy(x: Tensor) -> np.ndarray:
    x=x.detach().cpu().numpy()
    return x

#======================================================================================#

nbins = 250

def binomial_coeff(n,r):
    return math.factorial(n) / math.factorial(r) / math.factorial(n-r)

#===============================================================#
#              One-Body Density and Root-Mean-Square            #
#===============================================================#

sys.stdout.write("One-body Density: ")

# Calculate one-body density
x = configurations[:,:,:,0].flatten().detach().cpu().numpy() #modified

n, bins = np.histogram(x, bins=nbins, range=(xmin, xmax), density=True)

density_xx = bins[:-1] + np.diff(bins)/2. #grid of x in which the density is evaluated

density_psi = nfermions*n #density of the system

# Calculate RMS radius
rms = np.sum(density_psi * density_xx**2) / np.sum(density_psi)

sys.stdout.write("DONE\n")

#===============================================================#
#                     One-Body Density Matrix                   #
#===============================================================#

sys.stdout.write("One-body Density Matrix: ")

# Prepare OBDM data
xx = xdata.flatten().cpu().detach().numpy()
yy = ydata.flatten().cpu().detach().numpy()
p = nfermions * zdata.flatten().cpu().detach().numpy()

clamp = np.abs(np.array([xx.min(), xx.max(), yy.min(), yy.max()])).max()
data_range = [[-clamp,clamp],[-clamp,clamp]]

p = np.nan_to_num(p, nan=0.)

# Calculate OBDM
h_obdm, xedges_obdm, yedges_obdm = np.histogram2d(xx,yy,
                                                  bins=[nbins,nbins], range=data_range,
                                                  weights=p, density=True)
rho_matrix = nfermions * h_obdm / np.trace(h_obdm) #trace norm the histogram

sys.stdout.write("DONE\n")

#===============================================================#
#                      Occupation Numbers                       #
#===============================================================#

sys.stdout.write("Occupation Numbers: ")
# Diagonalize OBDM to get natural orbitals
eigenvalues, eigenvectors = np.linalg.eigh(rho_matrix) #diagonalize OBDM
eigen_idx = np.argsort(eigenvalues)[::-1] #sort

sorted_eigenvalues = eigenvalues[eigen_idx]
sorted_eigenvectors = eigenvectors[:, eigen_idx]

sys.stdout.write("DONE\n")

#===============================================================#
#              Two-Body Density (Pair-correlation)              #
#===============================================================#
sys.stdout.write("Two-body Density: ")
# Calculate pair correlation function
xxdata = configurations[:,:,:2,0].reshape(-1, 2).cpu().detach().numpy() #modified

bin_width = (xmax-xmin)/nbins
weight = (binomial_coeff(nfermions, 2) / bin_width**2) * np.ones_like(xxdata[:,0]) / xxdata.size

h_tbd, xedges_tbd, yedges_tbd = np.histogram2d(xxdata[:,0], xxdata[:,1],
                                               bins=[nbins, nbins], weights=weight,
                                               range=[[xmin, xmax],[xmin, xmax]],
                                               density=False)
sys.stdout.write("DONE\n")

#===============================================================#
#                         Save the data                         #
#===============================================================#
sys.stdout.write("Saving file: ")
# Save all calculated quantities
data = {'nbins':nbins,
        'V0':V0,
        'rms':rms, #root-mean-square
        'density_xx':density_xx, #density grid
        'density_psi':density_psi, #density values
        'h_obdm':h_obdm, #one-body density matrix
        'xedges_obdm':xedges_obdm, #x position of the one-body density matrix
        'yedges_obdm':yedges_obdm, #y position of the one-body density matrix
        'h_tbd':h_tbd, #two-body density matrix
        'xedges_tbd':xedges_tbd, #x position of the two-body density matrix
        'yedges_tbd':yedges_tbd, #y position of the two-body density matrix
        'eigenvalues':sorted_eigenvalues, #eigenvalues of the one-body density matrix
        'eigenvectors':sorted_eigenvectors} #eigenvectors of the one-body density matrix
np.savez_compressed(analysis_datapath, **data) #save
sys.stdout.write("DONE\n")