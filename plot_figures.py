'''This code plots the figures given '''
import torch
from torch import nn
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.colors as colors

import colorcet as cc

import argparse
from src.Pretraining import HermitePolynomialMatrixND
from scipy.ndimage import rotate

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
parser.add_argument("-Dim", "--dimensions", type=int,   default=2,     help="Number of dimensions of the physical system")
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

analysis_datapath = "analysis/PHYS_A%02i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_V%4.4e_S%4.4e_%s_PT_%s_device_%s_dtype_%s_dim_%02i.npz" % \
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
    plt.pcolormesh(X, Y, proj_xy.T, cmap='viridis')
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
    density_grid = loaded['density_grid'] #shape N^D
    density_edges = loaded['density_edges'] #list of edge arrays
    Y, Z = density_edges[1], density_edges[2]  # Assuming 2D for simplicity
    proj_yz = density_grid.sum(axis=(0,) + tuple(range(3, dimensions))) \
              if dimensions > 2 else density_grid
    plt.pcolormesh(Y, Z, proj_yz.T, cmap='viridis')
    plt.gca().set_aspect('equal')
    plt.colorbar()
    plt.title(f'Density n(y,z) (projected) for {nfermions} fermions')
    plt.xlabel('y')
    plt.ylabel('z')
    plt.show()

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
####################################################################### Comparison with the analytical density ##########################################################################################
###########################################################################################################################################################################
@torch.no_grad()
def analytic_density_1d_from_hpm(xgrid, nfermions, device="cpu"):
    """
    Devuelve n_ana(x) evaluada en xgrid (shape = (Nx,))
    usando HermitePolynomialMatrixND con Dim = 1.
    """
    hpm = HermitePolynomialMatrixND(num_particles=nfermions, Dim=1).to(device)
    hpm.eval()

    # xgrid → tensor [1, Nx, 1]  (nwalkers=1, A=Nx, Dim=1)
    coords = torch.tensor(xgrid, device=device).unsqueeze(0).unsqueeze(-1)

    # Salida HPM: [1, 1, Nx, N]  →  suma |φ|² sobre el último eje
    n_ana = hpm(coords).pow(2).sum(dim=-1).squeeze()      # shape (Nx,)
    return n_ana.cpu().numpy()


@torch.no_grad()
def analytic_density_from_hpm(edges, nfermions, device="cpu"):
    """
    Returns an analytic density array with the SAME shape as `density_grid`
    using HermitePolynomialMatrixND.  Works for D = 1, 2 or 3.

    edges  : list of bin-edge arrays    (same object as loaded['density_edges'])
    nfermions : number of particles N
    device : 'cpu' | 'cuda'
    """
    Dim = len(edges)
    hpm = HermitePolynomialMatrixND(num_particles=nfermions, Dim=Dim).to(device)
    hpm.eval()

    # --- 1. Mesh of bin centres --------------------------------------------
    centres = [e[:-1] + np.diff(e)/2 for e in edges]          # list (Nx, Ny, ...)
    mesh    = np.meshgrid(*centres, indexing="ij")
    grid_shape = mesh[0].shape                                # e.g. (Nx, Ny, Nz)
    Npts = np.prod(grid_shape)

    # --- 2. Flatten coordinates into [1, Npts, Dim] ------------------------
    coords_np = np.stack([m.ravel() for m in mesh], axis=1)   # (Npts, Dim)
    coords    = torch.tensor(coords_np, device=device).unsqueeze(0)  # [1, Npts, Dim]

    # --- 3. Forward pass: orbitals → density -------------------------------
    # hpm output: [nwalkers=1, 1, A=Npts, num_particles]
    orbital_mat = hpm(coords)                     # shape (1, 1, Npts, N)
    # sum |φ|² over the last axis (num_particles)
    n_ana_flat  = orbital_mat.pow(2).sum(dim=-1).squeeze(0).squeeze(0)  # (Npts,)

    # --- 4. Reshape back to Cartesian grid ---------------------------------
    return n_ana_flat.cpu().numpy().reshape(grid_shape)

if dimensions == 1:
    n_ana = analytic_density_1d_from_hpm(density_xx, nfermions, device=device)
else:
    n_ana = analytic_density_from_hpm(density_edges, nfermions, device=device)




##########Rotation
def best_rotation_2d(arr_nqs, arr_ref, axes=(0, 1), n_angles=181):
    """
    Find the in-plane rotation angle θ (degrees) that minimises
    ‖nqs(θ) – ref‖₂.  Returns (θ_best, rotated_arr).
    """
    thetas = np.linspace(0, 360, n_angles, endpoint=False, dtype=float)
    best_theta, best_err = 0.0, np.inf
    best_rot = None
    for th in thetas:
        rot = rotate(arr_nqs, angle=th, axes=axes, reshape=False,
                     order=1, mode='nearest')
        err = np.linalg.norm(rot - arr_ref)
        if err < best_err:
            best_theta, best_err, best_rot = th, err, rot
    return best_theta, best_rot

def rotate_3d(arr, angles_deg):
    """
    Apply a Z-Y-Z Euler rotation (γ, β, α) to a 3-D array.
    angles_deg = (alpha, beta, gamma) in degrees.
    """
    a, b, g = angles_deg            # α, β, γ
    out = rotate(arr, g, axes=(0,1), reshape=False, order=1, mode='nearest')  # Z
    out = rotate(out, b, axes=(0,2), reshape=False, order=1, mode='nearest')  # Y
    out = rotate(out, a, axes=(0,1), reshape=False, order=1, mode='nearest')  # Z
    return out

def apply_rotation_3d_and_error(density_nqs, density_ref,
                                euler_deg, norm="L2",
                                mode="nearest", order=1):
    """
    Parameters
    ----------
    density_nqs : np.ndarray (Nx, Ny, Nz)
        Raw NQS density grid.
    density_ref : np.ndarray (Nx, Ny, Nz)
        Analytic/reference density grid.
    euler_deg   : tuple(float, float, float)
        (α, β, γ) in degrees ‒ Z-Y-Z convention.
    norm        : {"L2", "L1"}
        Which error metric to return.
    mode, order : kwargs passed to scipy.ndimage.rotate.

    Returns
    -------
    err_value : float
        Chosen norm of (rotated − reference).
    density_rot : np.ndarray
        Rotated NQS density (same shape as input).
    """
    assert len(density_nqs.shape) == 3, "This helper is for 3-D grids only."
    density_rot = rotate_3d(density_nqs, euler_deg)

    diff = density_rot - density_ref
    if norm.upper() == "L2":
        err_value = np.linalg.norm(diff)
    elif norm.upper() == "L1":
        err_value = np.sum(np.abs(diff))
    else:
        raise ValueError("norm must be 'L1' or 'L2'.")

    return euler_deg, density_rot

def best_rotation_3d(arr_nqs, arr_ref, coarse_step=30):
    """
    Very coarse grid search over Euler angles (α, β, γ) ∈ [0,360) × [0,180] × [0,360).
    Returns (angles_best, rotated_arr).  Refine manually if necessary.
    """
    alphas = np.arange(0, 360, coarse_step)
    betas  = np.arange(0, 181, coarse_step)         # [0,180]
    gammas = np.arange(0, 360, coarse_step)
    best_angles, best_err, best_rot = None, np.inf, None
    for a in alphas:
        for b in betas:
            for g in gammas:
                rot = rotate_3d(arr_nqs, (a, b, g))
                err = np.linalg.norm(rot - arr_ref)
                if err < best_err:
                    best_angles, best_err, best_rot = (a, b, g), err, rot
    return best_angles, best_rot

'''Best Euler angles (α,β,γ) ≈ (0, 90, 210) for 2 fermions'''
# -----------------------------------------------------------------
best2ferm=(0,90,210)
best3ferm=(300, 90, 0)
if dimensions == 2:
    # full XY arrays
    theta, density_grid_rot = best_rotation_2d(density_grid, n_ana.T)
    print(f"[INFO] Best in-plane rotation: θ ≈ {theta:.1f}°")

elif dimensions == 3:
    # Work on the full 3-D grids
    print(f"[INFO] Finding best rotation for 3D density grid (shape {density_grid.shape})...")
    angles, density_grid_rot = apply_rotation_3d_and_error(density_grid, n_ana.T, euler_deg=best3ferm)
    print(f"[INFO] Best Euler angles (α,β,γ) ≈ {angles}")

    # If you also keep the projected maps, update them:
    density_grid = density_grid_rot
##########Plot

if dimensions == 1:
    print("Analytical density shape:", n_ana.shape)
    plt.figure()
    plt.title('Analytical n(x) for %i fermions' % (nfermions))
    plt.xlabel('Position, x')
    plt.ylabel('Density, n(x)')
    #plt.ylim(0, 2)
    plt.plot(density_xx, n_ana.T)
    plt.show()
# ---------- |Δn| visualisation ----------------------------------------------
if dimensions == 1:
    # nqs density in 'density_psi', analytic in 'n_ana'  (shape = (Nx,))
    delta_abs = np.abs(density_psi - n_ana.T)

    plt.figure()
    plt.title(r'Absolute difference $|\,\Delta n(x)\,|$')
    plt.xlabel('Position, x')
    plt.ylabel(r'$|\,\Delta n(x)\,|$')
    plt.plot(density_xx, delta_abs, color='tab:red')
    plt.axhline(0, color='k', lw=0.8)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------
elif dimensions == 2:
    X, Y = density_edges[0], density_edges[1]          # bin edges
    plt.figure()
    plt.pcolormesh(X, Y, n_ana, cmap='viridis')
    plt.gca().set_aspect('equal')
    plt.colorbar()
    plt.title(f'Density n(x,y) (projected) for {nfermions} fermions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    # densities have shape (Nx, Ny)
    delta_grid = np.abs(density_grid_rot.T - n_ana)          # full 2-D array

    X, Y = density_edges[0], density_edges[1]          # bin edges
    plt.figure()
    plt.pcolormesh(X, Y, delta_grid, cmap='inferno')
    plt.gca().set_aspect('equal')
    plt.colorbar(label=r'$|\,\Delta n(x,y)\,|$')
    plt.title(fr'Absolute difference $|\,\Delta n(x,y)\,|$ '
              f'for {nfermions} fermions')
    plt.xlabel('x');  plt.ylabel('y')
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------
elif dimensions == 3:
    # densities have shape (Nx, Ny, Nz)
    delta_grid = np.abs(density_grid.T - n_ana)

    # ---- XY projection -----------------------------------------------------
    X, Y = density_edges[0], density_edges[1]
    proj_xy = delta_grid.sum(axis=2) * np.diff(density_edges[2])[0]  # integrate z

    plt.figure()
    plt.pcolormesh(X, Y, proj_xy.T, cmap='inferno')
    plt.gca().set_aspect('equal')
    plt.colorbar(label=r'$|\,\Delta n(x,y)\,|$ (integrated over z)')
    plt.title(fr'$|\,\Delta n(x,y)\,|$  (proj.) for {nfermions} fermions')
    plt.xlabel('x');  plt.ylabel('y')
    plt.tight_layout()
    plt.show()

    # ---- YZ projection -----------------------------------------------------
    Y, Z = density_edges[1], density_edges[2]
    proj_yz = delta_grid.sum(axis=0) * np.diff(density_edges[0])[0]  # integrate x

    plt.figure()
    plt.pcolormesh(Y, Z, proj_yz.T, cmap='inferno')
    plt.gca().set_aspect('equal')
    plt.colorbar(label=r'$|\,\Delta n(y,z)\,|$ (integrated over x)')
    plt.title(fr'$|\,\Delta n(y,z)\,|$  (proj.) for {nfermions} fermions')
    plt.xlabel('y');  plt.ylabel('z')
    plt.tight_layout()
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