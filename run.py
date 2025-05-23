##################################################################################################################################################
#####                                  Setting the parser to configurate how the code is run from the terminal                               #####
##################################################################################################################################################
'''This parser is used to run the main code. And allows to run the code with different parameters using the terminal.
e.g python run.py --num_fermions 4 --num_hidden 128 --V0 0.5'''
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(prog="SpinlessFermions",
                                 usage='%(prog)s [options]',
                                 description="A Neural Quantum State (NQS) solution to one-dimensional fermions interacting in a Harmonic trap",
                                 epilog="and fin")
#https://stackoverflow.com/questions/14117415/in-python-using-argparse-allow-only-positive-integers/14117567

parser.add_argument("-N", "--num_fermions", type=int,   default=2,     help="Number of fermions in physical system")
parser.add_argument("-H", "--num_hidden",   type=int,   default=64,    help="Number of hidden neurons per layer")
parser.add_argument("-L", "--num_layers",   type=int,   default=2,     help="Number of layers within the network")
parser.add_argument("-D", "--num_dets",     type=int,   default=1,     help="Number of determinants within the network's final layer")
parser.add_argument("-V", "--V0",           type=float, default=0.,    help="Interaction strength (in harmonic units)")
parser.add_argument("-S", "--sigma0",       type=float, default=0.5,   help="Interaction distance (in harmonic units")
parser.add_argument("--preepochs",          type=int,   default=1000, help="Number of pre-epochs for the pretraining phase")
parser.add_argument("--epochs",             type=int,   default=10000, help="Number of epochs for the energy minimisation phase")
parser.add_argument("-C", "--chunks",       type=int,   default=1,     help="Number of chunks for vectorized operations")
parser.add_argument("--dtype",              type=str,   default='float32',      help='Default dtype')
parser.add_argument("-Dim", "--dimensions", type=int,   default=2,     help="Number of dimensions of the physical system")
parser.add_argument("--debug", type=bool,  default=True, help="Debug mode")

args = parser.parse_args()

###############################################################################################################################################
#####                                           Setting the NQS model and the Hamiltonian                                                 #####
###############################################################################################################################################
import torch
from torch import nn, Tensor
import numpy as np

import os, sys, time

torch.manual_seed(0) #set seed
torch.set_printoptions(4) #set the digits of precision
torch.backends.cudnn.benchmark=True #this setting enables automatic tuning of CUDA kernels to find the most efficient algorithm for your specific hardware

#set the default dtype
if(args.dtype == 'float32'):
    torch.set_default_dtype(torch.float32)
elif(args.dtype == 'float64'):
    torch.set_default_dtype(torch.float64)
else:
    raise NameError(f"Unknown dtype: {args.dtype} selected!")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #set device
dtype = str(torch.get_default_dtype()).split('.')[-1] #get dtype

DIR='./'
sys.path.append(DIR+"src/")

from src.Models import vLogHarmonicNet
from src.Samplers import MetropolisHastings
from src.Hamiltonian import HarmonicOscillatorWithInteractionD as HOwD
from src.Pretraining import HermitePolynomialMatrix, HermitePolynomialMatrixND 

from src.utils import load_dataframe, load_model, count_parameters, get_groundstate
from src.utils import get_params, sync_time, clip, calc_pretraining_loss

nfermions = args.num_fermions #number of input nodes
num_hidden = args.num_hidden  #number of hidden nodes per layer
num_layers = args.num_layers  #number of layers in network
num_dets = args.num_dets      #number of determinants (accepts arb. value)
func = nn.Tanh()  #activation function between layers
pretrain = True   #command to pretrain the network
debug = args.debug #debug mode

nwalkers=4096 #number of walkers in the Metropolis
n_sweeps=10 #n_discard
std=1.#0.02#1.
target_acceptance=0.5 #target acceptance rate 0.5

V0 = args.V0 #interaction strength
sigma0 = args.sigma0 #interaction distance

pt_save_every_ith=1000 #save every ith epoch in the pretraining
em_save_every_ith=1000 #save every ith epoch in the energy minimisation

nchunks=1 #only one chunk is used so it is not implemented in the code

preepochs=args.preepochs #number of pre-epochs
epochs=args.epochs #number of epochs
dimensions=args.dimensions #number of dimensions



#define the network
#torch.manual_seed(42)
net = vLogHarmonicNet(num_input=nfermions,
                      num_hidden=num_hidden,
                      num_layers=num_layers,
                      num_dets=num_dets,
                      func=func,
                      pretrain=pretrain,
                      Dim=dimensions) #modified
net=net.to(device)

#set the sampler, returns the chains and the log_prob
#torch.manual_seed(42)
sampler = MetropolisHastings(network=net,
                             dof=nfermions,
                             nwalkers=nwalkers,
                             target_acceptance=target_acceptance,
                             dim=dimensions) #modified



#compute the local energy
calc_elocal = HOwD(net=net, V0=V0, sigma0=sigma0, nchunks=nchunks, dimensions= dimensions)

HO = HermitePolynomialMatrixND(num_particles=nfermions,Dim=dimensions) #target wavefunction for pretraining #cambiar estos #modified

#set the optimizer
optim = torch.optim.Adam(params=net.parameters(), lr=1e-4) 


gs_CI = get_groundstate(A=nfermions, V0=V0, datapath="groundstate/")#get the analytical groundstate from the groundstate folder

print("Network     | A: %4i | H: %4i | L: %4i | D: %4i | Dimensions: %4i" % (nfermions, num_hidden, num_layers, num_dets, dimensions))
print("Sampler     | B: %4i | T: %4i | std: %4.2f | targ: %s" % (nwalkers, n_sweeps, std, str(target_acceptance)))
print("Hamiltonian | V0: %4.2f | S0: %4.2f" % (V0, sigma0))
print("Pre-epochs: | %6i" % (preepochs))
print("Epochs:     | %6i" % (epochs))
print("Number of parameters: %8i\n" % (count_parameters(net)))

###############################################################################################################################################
#####                                                          DEBUGGING                                                                  #####
###############################################################################################################################################
if debug==True:
    def dummy_logpdf(x):
        return -x.pow(2).sum(dim=(-1, -2))

    import matplotlib.pyplot as plt

    #sampler.log_pdf = dummy_logpdf

    x, _ = sampler(n_sweeps=1)  # x.shape = [nwalkers, A, D]

    # Escoge un único fermión (por ejemplo, el primero: índice 0)
    coords = x[:, 0, :]        # [nwalkers, 2] → coordenadas (x, y) del primer fermión
    x1, y1 = coords[:, 0].cpu(), coords[:, 1].cpu()

    plt.figure(figsize=(6,6))
    plt.scatter(x1, y1, s=3, alpha=0.4)
    plt.xlabel(r'$x$'); plt.ylabel(r'$y$')
    plt.title('Distribución inicial de walkers – primer fermión')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()




###############################################################################################################################################
#####                                           PRE-TRAINING LOOP                                                                         #####
###############################################################################################################################################

model_path_pt = DIR+"results/pretrain/checkpoints/A%02i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_%s_PT_%s_device_%s_dtype_%s_dim_%02i_chkp.pt" % \
                 (nfermions, num_hidden, num_layers, num_dets, func.__class__.__name__, nwalkers, preepochs, \
                  optim.__class__.__name__, True, device, dtype, dimensions)
filename_pt = DIR+"results/pretrain/data/A%02i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_%s_PT_%s_device_%s_dtype_%s_dim_%02i.csv" % \
                 (nfermions, num_hidden, num_layers, num_dets, func.__class__.__name__, nwalkers, preepochs, \
                  optim.__class__.__name__, True, device, dtype,dimensions)

net.pretrain = True #set the pretraining network mode

#If computed, we load the pretrained model and the previously computed data
writer_pt = load_dataframe(filename_pt)
output_dict = load_model(model_path=model_path_pt, device=device, net=net, optim=optim, sampler=sampler)

start=output_dict['start'] #unpack dict with the trained model
net=output_dict['net']
optim=output_dict['optim']
sampler=output_dict['sampler']



#print(torch.rand(1))
#torch.manual_seed(42)
#x, _ = sampler(n_sweeps=1)#n_sweeps)
#print(x.shape)
#print(x.squeeze(-1))
#print('hello')
#Pre-training
# #print(torch.rand(1))
for preepoch in range(start, preepochs+1):     #in case it has already been trained start=preepochs+1
    stats={}
    
    start=sync_time()

    x, _ = sampler(n_sweeps=n_sweeps) #sample illustrating the wavefunction of the network
    
    network_orbitals = net(x) #wave function of the network
    #print(network_orbitals.shape)
    target_orbitals = HO(x) #target wave function for the pretraining computed analytically
    
    mean_preloss, stddev_preloss = calc_pretraining_loss(network_orbitals, target_orbitals) #loss function and its standard deviation

    #optimization step
    optim.zero_grad()
    mean_preloss.backward()  #con cpu me rompe por aqui
    optim.step()

    end = sync_time()

    #save epoch info
    stats['epoch'] = [preepoch] 
    stats['loss_mean'] = mean_preloss.item()
    stats['loss_std'] = (stddev_preloss.pow(2) / nwalkers).sqrt().item()
    stats['proposal_width'] = sampler.sigma
    stats['acceptance_rate'] = sampler.acceptance_rate
    
    stats['walltime'] = end-start
    
    #and write it into a .csv file
    writer_pt(stats) #push data to Writer

    #if its in the corresponding epoch, save the model
    if(preepoch % pt_save_every_ith == 0):
        torch.save({'epoch':preepoch,
                    'model_state_dict':net.state_dict(),
                    'optim_state_dict':optim.state_dict(),
                    'loss':mean_preloss.item(),
                    'chains':sampler.chains.detach(),
                    'log_prob':sampler.log_prob.detach(),
                    'sigma':sampler.sigma.item()},
                    model_path_pt)
        writer_pt.write_to_file(filename_pt)
        

    sys.stdout.write("Epoch: %6i | Loss: %6.4f +/- %6.4f | Walltime: %4.2e (s)      \r" % (preepoch, mean_preloss, stddev_preloss, end-start))
    sys.stdout.flush()

print("\n")

#####################################################################################################################################
#####                                                   Debugging                                                                    #####
##########################################################################################################################################
# ── Plot de validación tras el pre-training (solo 1-D) ────────────────────────
if debug==True:
    if dimensions == 1:
        import matplotlib.pyplot as plt

        net.eval();  net.pretrain = True           # red en modo pre-training

        with torch.no_grad():
            # Usa la misma variable n_sweeps definida al principio (100 por defecto)
            x, _ = sampler(n_sweeps=n_sweeps)      # ← aquí metemos n_sweeps
            phi_N = net(x)                         # [B,1,A,A]
            phi_H = HO(x)

            psi_net = torch.linalg.det(phi_N.squeeze(1))
            psi_ref = torch.linalg.det(phi_H.squeeze(1))

        plt.figure(figsize=(6,6))
        plt.scatter(psi_ref.cpu(), psi_net.cpu(), alpha=0.4)
        lims = [psi_ref.min().item(), psi_ref.max().item()]
        plt.plot(lims, lims, color='k', lw=1)      # línea identidad
        plt.xlabel(r'$\psi_{\mathrm{ref}}$  (Hermite)')
        plt.ylabel(r'$\psi_{\mathrm{net}}$')
        plt.title(f'Pre-training check (1-D) –  {n_sweeps} sweeps')
        plt.tight_layout()
        plt.show()

        net.eval()
        net.pretrain = False          # estamos en fase de minimizado de energía

        with torch.no_grad():
            x, _ = sampler(n_sweeps=n_sweeps)   # x: [B, 2, 1]
            sign, logabs = net(x)               # salida many-body  [B]
            psi_abs = torch.exp(logabs)         # |ψ|              [B]

            # ---- eje horizontal: coordenada del primer fermión ----
            x1 = x[:, 0, 0]          # [B]

            # ---- ordena para una curva continua ----
            idx      = torch.argsort(x1)
            x_sorted = x1[idx].cpu()
            psi_sorted = psi_abs[idx].cpu()   # usa .pow(2) para densidad

            # ---- gráfico ----
            plt.figure(figsize=(7,4))
            plt.plot(x_sorted, psi_sorted, '.', ms=3)
            plt.xlabel(r'$x_1$   ')
            plt.ylabel(r'$|\psi_{\mathrm{net}}|$')
            plt.title('Wave function 1D of fermions after pretraining')
            plt.tight_layout()
            plt.show()

        
    # ── DEBUG VIS – caso bidimensional (A = 2, Dim = 2) ────────────────────────
    if dimensions == 2:
        import matplotlib.pyplot as plt
        net.eval()

        # ==============================================================
        # 1.  Chequeo de pre-training (ψ_red vs ψ_ref)  – igual que en 1-D
        # ==============================================================
        net.pretrain = True
        with torch.no_grad():
            x, _ = sampler(n_sweeps=n_sweeps)      # x: [B, 2, 2]
            phi_N = net(x)                         # [B,1,2,2]
            phi_H = HO(x)                          # analítico

            psi_net = torch.linalg.det(phi_N.squeeze(1))
            psi_ref = torch.linalg.det(phi_H.squeeze(1))

        plt.figure(figsize=(6,6))
        plt.scatter(psi_ref.cpu(), psi_net.cpu(), alpha=0.4)
        lims = [psi_ref.min().item(), psi_ref.max().item()]
        plt.plot(lims, lims, 'k', lw=1)
        plt.xlabel(r'$\psi_{\mathrm{ref}}$')
        plt.ylabel(r'$\psi_{\mathrm{net}}$')
        plt.title(f'Pre-training check (2-D)  – {n_sweeps} sweeps')
        plt.tight_layout();  plt.show()

        # ==============================================================
        # 2.  Densidad |ψ|² en el plano (x1x, x1y)  después del entrenamiento
        # ==============================================================
        print("x shape:", x.shape)           #  ¿[B, 2, 2]?
        print("ejemplo walker 0:", x[0])     #  [[x1x, x1y],
                                            #   [x2x, x2y]]
        diff_xy = (x[:,:,0] - x[:,:,1]).abs().max()
        print("máx |x - y| en TODO el batch:", diff_xy.item())



        net.pretrain = False
        with torch.no_grad():
            x, _ = sampler(n_sweeps=n_sweeps)      # [B,2,2]
            sign, logabs = net(x)                  # [B]
            prob = torch.exp(2*logabs).cpu()       # |ψ|²

        # Coordenadas del PRIMER fermión: (x, y)
        # ------------------------------------------------------------
        # 1)  Extrae SOLO la coordenada x e y del MISMO fermión
        # ------------------------------------------------------------
        x1 = x[:, 0, 0].cpu().numpy()   # primer fermión, comp. x
        y1 = x[:, 0, 1].cpu().numpy()   # primer fermión, comp. y

        # rápido sanity-check
        print("\nParejas que voy a graficar:")
        for i in range(5):
            print(f"{i}: ({x1[i]:+.3f}, {y1[i]:+.3f})  vs tensor {x[i,0,:].tolist()}")
        print("pares (x1,y1) :", list(zip(x1[:5], y1[:5])))

        # ------------------------------------------------------------
        # 2)  Plot con aspecto cuadrado
        # ------------------------------------------------------------
        cvals = np.log10(prob.cpu().numpy() + 1e-12)

        fig, ax = plt.subplots(figsize=(6,6))
        hb = ax.hexbin(x1, y1, C=cvals,
                       gridsize=70, cmap='magma',
                       reduce_C_function=np.mean)

        ax.set_aspect('equal', adjustable='box')
        R = max(abs(x1).max(), abs(y1).max())
        ax.set_xlim(-R, R); ax.set_ylim(-R, R)

        cb = fig.colorbar(hb, ax=ax, label=r'$\log_{10}|\psi|^{2}$')
        ax.set_xlabel(r'$x_{1,x}$')
        ax.set_ylabel(r'$x_{1,y}$')
        ax.set_title(r'Densidad $|\psi|^{2}$ – primer fermión (Dim = 2)')
        plt.tight_layout(); plt.show()
        







###############################################################################################################################################
#####                                           ENERGY-MINIMISATION LOOP                                                                  #####
###############################################################################################################################################

net.pretrain = False #this argument is needed in order to change the network into the energy minimisation mode
optim = torch.optim.Adam(params=net.parameters(), lr=1e-4) #new optimizer

model_path = DIR+"results/energy/checkpoints/A%02i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_V%4.4e_S%4.4e_%s_PT_%s_device_%s_dtype_%s_dim_%02i_chkp.pt" % \
                (nfermions, num_hidden, num_layers, num_dets, func.__class__.__name__, nwalkers, preepochs, V0, sigma0, \
                 optim.__class__.__name__, False, device, dtype,dimensions)
filename = DIR+"results/energy/data/A%02i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_V%4.4e_S%4.4e_%s_PT_%s_device_%s_dtype_%s_dim_%02i.csv" % \
                (nfermions, num_hidden, num_layers, num_dets, func.__class__.__name__, nwalkers, preepochs, V0, sigma0, \
                 optim.__class__.__name__, False, device, dtype,dimensions)

#If computed, we load the pretrained model and the previously computed data
writer = load_dataframe(filename)
output_dict = load_model(model_path=model_path, device=device, net=net, optim=optim, sampler=sampler)

start=output_dict['start'] #unpack dict with the trained model
net=output_dict['net']
optim=output_dict['optim']
sampler=output_dict['sampler']

#Energy Minimisation
progress_bar = tqdm(range(start, epochs+1), total=epochs+1, initial=start, desc="Training", dynamic_ncols=True)

for epoch in progress_bar:
#for epoch in tqdm(range(start, epochs+1), total=epochs+1, initial=start, desc="Training", dynamic_ncols=True):    #in case it has already been trained start=epochs+1
    stats={}

    start=sync_time()
    
    x, _ = sampler(n_sweeps) #sample illustrating the wavefunction of the network
    #print(x.shape)
    elocal, _kin, _potential, _inter = calc_elocal(x) #compute the local energy
    elocal_clipped = clip(elocal, clip_factor=5) #clip the local energy
    print(elocal)
    print(elocal_clipped)
    _, logabs = net(x) #compute the global logabs values of Generalised Slater Matrices of the network

    loss_elocal = 2.*((elocal - torch.mean(elocal)).detach() * logabs)
    
    with torch.no_grad():
        energy_var, energy_mean = torch.var_mean(elocal, unbiased=True) #compute the variance and the mean of the local energy
        energy_std = (energy_var / nwalkers).sqrt() #compute the standard deviation of the local energy

    loss=torch.mean(loss_elocal)    #loss function
    
    #optimization step
    optim.zero_grad()
    loss.backward()  #populates leafs with grads
    optim.step()

    end = sync_time()

    #save epoch info
    stats['epoch'] = [epoch] #must pass index
    stats['loss'] = loss.item()
    stats['energy_mean'] = energy_mean.item()
    stats['energy_std'] = energy_std.item()
    stats['CI'] = gs_CI
    stats['proposal_width'] = sampler.sigma.item()
    stats['acceptance_rate'] = sampler.acceptance_rate

    stats['walltime'] = end-start

    writer(stats) #push data to Writer

    if epoch % 100 == 0:          # every 100 sampler moves
        print(f"step {epoch:5d}  "
            f"⟨E⟩={elocal.mean():9.3e}   "
            f"min/max={elocal.min():9.3e}/{elocal.max():9.3e}   "
            f"σ={elocal.std():9.3e}")

    #if its in the corresponding epoch, save the model
    if(epoch % em_save_every_ith == 0):
        torch.save({'epoch':epoch,
                    'model_state_dict':net.state_dict(),
                    'optim_state_dict':optim.state_dict(),
                    'loss':loss,
                    'energy':energy_mean,
                    'energy_std':energy_var.sqrt(),
                    'chains':sampler.chains.detach(),
                    'log_prob':sampler.log_prob.detach(),
                    'sigma':sampler.sigma},
                    model_path)
        writer.write_to_file(filename)

    #sys.stdout.write("Epoch: %6i | Energy: %6.4f +/- %6.4f | CI: %6.4f | Walltime: %4.2e (s)        \r" % (epoch, energy_mean, energy_std, gs_CI, end-start))
    #sys.stdout.flush()
    #tqdm.write("Epoch: %6i | Energy: %6.4f +/- %6.4f | CI: %6.4f | Walltime: %4.2e (s)" %
    #       (epoch, energy_mean, energy_std, gs_CI, end - start))
    progress_bar.set_postfix({
        "Epoch": f"{epoch:6d}",
        "Energy": f"{energy_mean:.5f}",
        "+/-": f"{energy_std:.5f}",
        "CI": f"{gs_CI:.4f}",
        "t(s)": f"{(end - start):.2e}"
    })



from src.utils import generate_final_energy, round_to_err, str_with_err

#####################################################################################################################################
#####                                                   Debugging                                                                    #####
##########################################################################################################################################
# ── Plot de validación tras el training (solo 1-D) ────────────────────────
if debug==True:
    if dimensions == 1:
        import matplotlib.pyplot as plt

        net.eval();  net.pretrain = True           # red en modo pre-training

        with torch.no_grad():
            # Usa la misma variable n_sweeps definida al principio (100 por defecto)
            x, _ = sampler(n_sweeps=n_sweeps)      # ← aquí metemos n_sweeps
            phi_N = net(x)                         # [B,1,A,A]
            phi_H = HO(x)

            psi_net = torch.linalg.det(phi_N.squeeze(1))
            psi_ref = torch.linalg.det(phi_H.squeeze(1))

        plt.figure(figsize=(6,6))
        plt.scatter(psi_ref.cpu(), psi_net.cpu(), alpha=0.4)
        lims = [psi_ref.min().item(), psi_ref.max().item()]
        plt.plot(lims, lims, color='k', lw=1)      # línea identidad
        plt.xlabel(r'$\psi_{\mathrm{ref}}$  (Hermite)')
        plt.ylabel(r'$\psi_{\mathrm{net}}$')
        plt.title(f'Post-training check (1-D) –  {n_sweeps} sweeps')
        plt.tight_layout()
        plt.show()

        net.eval()
        net.pretrain = False          # estamos en fase de minimizado de energía

        with torch.no_grad():
            x, _ = sampler(n_sweeps=n_sweeps)   # x: [B, 2, 1]
            sign, logabs = net(x)               # salida many-body  [B]
            psi_abs = torch.exp(logabs)         # |ψ|              [B]

            # ---- eje horizontal: coordenada del primer fermión ----
            x1 = x[:, 0, 0]          # [B]

            # ---- ordena para una curva continua ----
            idx      = torch.argsort(x1)
            x_sorted = x1[idx].cpu()
            psi_sorted = psi_abs[idx].cpu()   # usa .pow(2) para densidad

            # ---- gráfico ----
            plt.figure(figsize=(7,4))
            plt.plot(x_sorted, psi_sorted, '.', ms=3)
            plt.xlabel(r'$x_1$')
            plt.ylabel(r'$|\psi_{\mathrm{net}}|$')
            plt.title('Wave function 1D of fermions after training')
            plt.tight_layout()
            plt.show()

###############################################################################################################################################
#####                                               FINAL MODEL EVALUATION                                                                #####
###############################################################################################################################################

n_batches = 10_000

#In order to compute the final energy, we use the trained neural network (net) to generate many-body quantum configurations using Metropolis-Hastings sampling
energy_stats = generate_final_energy(calc_elocal=calc_elocal,
                                        sampler=sampler,
                                        n_batches=n_batches,
                                        chunk_size=None, #full-batch vectorization
                                        n_sweeps=1,     #10 is fine, 400 is too much 
                                        storage_device=torch.device('cpu')) #store on cpu to save memory for GPU
energy_mean=energy_stats['mean']
error_of_mean=energy_stats['error_of_mean']
batch_variance=energy_stats['batch_variance']
variance=energy_stats['variance']
R_hat=energy_stats['R_hat']
ESS=energy_stats['ESS']
tau_min=energy_stats['tau_min']
tau_avg=energy_stats['tau_avg']
tau_std=energy_stats['tau_std']
tau_max=energy_stats['tau_max']

energy_mean, error_of_mean = round_to_err(energy_mean.item(), error_of_mean.item())
energy_str = str_with_err(energy_mean, error_of_mean)
print(f"Energy: {energy_str} | R_hat: {R_hat:6.4f}")

final_energy_str = DIR+"results/final/FINAL_A%02i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_V%4.4e_S%4.4e_%s_PT_%s_device_%s_dtype_%s_dim_%02i.npz" % \
                (nfermions, num_hidden, num_layers, num_dets, func.__class__.__name__, nwalkers, preepochs, V0, sigma0, optim.__class__.__name__, False, 
device, dtype,dimensions)
print(f"Saving to {final_energy_str}")
data = {'energy_mean': energy_mean,
        'error_of_mean': error_of_mean,
        'energy_str':energy_str, #the energy mean with its error in a string version
        'batch_variance':batch_variance, #variance of the means across batches
        'variance':variance, #variance of the energy mean
        'R_hat':R_hat, #Gelman-Rubin diagnostic is a convergence diagnostic
        'ESS':ESS,#effective sample size
        'tau_min':tau_min, #minimum autocorrelation time across chains
        'tau_avg':tau_avg, #mean autocorrelation time
        'tau_std':tau_std, #standard deviation of the autocorrelation time
        'tau_max':tau_max, #maximum autocorrelation time
        'CI':gs_CI}
np.savez(final_energy_str, **data)
