#Pytorch package
import torch
from torch import nn, Tensor
import math
#Typecasting
from typing import Tuple, Callable

def rw_metropolis_kernel_2D(logpdf: Callable, position: Tensor, log_prob: Tensor, sigma: float, dim: int=1):
    #Shamelessly taken from https://rlouf.github.io/post/jax-random-walk-metropolis/
    
    proposal = position + sigma * torch.randn(position.shape, device=position.device)
    
    proposal_logprob = logpdf(proposal)
    log_uniform = torch.log(torch.rand(proposal_logprob.shape, device=position.device))
    accept = log_uniform < proposal_logprob - log_prob
    
    acceptance_rate = accept.float().mean()

    position = accept[:, None, None] * proposal + (~accept[:, None, None]) * position
    log_prob = torch.where(accept, proposal_logprob, log_prob)
    return position, log_prob, acceptance_rate



class MetropolisHastings_2D(nn.Module):

    def __init__(self, network: nn.Module, dof: int, nwalkers: int, target_acceptance: float, dim: int=1) -> None:
        super(MetropolisHastings_2D, self).__init__()

        # Set the parameters
        self.network = network #network to be trained
        self.dof = dof #number of fermions
        self.nwalkers = nwalkers #number of walkers
        self.target_acceptance = target_acceptance #target acceptance rate
        self.dim = dim #dimension of the system

        self.device = next(self.network.parameters()).device #device to run the simulation

        self.sigma = torch.tensor(1.0, device=self.device) #standard deviation of the proposal distribution 1.0
        self.acceptance_rate = torch.tensor(0.0, device=self.device) #acceptance rate
        #torch.manual_seed(42)
        self.chains = torch.randn(size=(self.nwalkers, self.dof, self.dim), #modified here 
                                        device=self.device,
                                        requires_grad=False) #isotropic initialisation of the points following the distribution.
        #print(f"Pos inicial {self.chains.squeeze(-1)}")
        _pretrain = self.network.pretrain
        if(_pretrain==True):
            self.network.pretrain=False  #inside the Metropolis we work with the original network
        self.log_prob = self.network(self.chains)[1].mul(2)
        self.network.pretrain=_pretrain

    def log_pdf(self, x: Tensor,alpha: float = 1.0) -> Tensor:
      '''2*Logarithm of the probability density function, and alpha reemphorces the curvature of the wavefunction'''
      _pretrain = self.network.pretrain
      if(_pretrain==True):
        self.network.pretrain=False
      _, logabs = self.network(x)
      self.network.pretrain=_pretrain
      return 2.*logabs*alpha

    def _update_sigma(self, acceptance_rate: Tensor) -> Tensor: 
        '''Update the standard deviation of the proposal distribution'''
        #Shamelessly taken from https://github.com/deepqmc/deepqmc
        scale_factor = self.target_acceptance / max(acceptance_rate, 0.05)
        return self.sigma / scale_factor
    
    def _initialize_walkers(self):
        """Inicializa los walkers desde una distribución normal. (Añadido para debugging)"""
        self.chains = torch.randn(size=(self.nwalkers, self.dof, self.dim),
                                  device=self.device,
                                  requires_grad=False)
        _pretrain = self.network.pretrain
        if _pretrain:
            self.network.pretrain = False
        self.log_prob = self.network(self.chains)[1].mul(2)
        self.network.pretrain = _pretrain

    def reset_walkers(self):
        """Reinicia los walkers para que empiecen desde cero. (Añadido para debugging)"""
        self._initialize_walkers()
        print("Walkers reseteados a su estado inicial.")

    @torch.no_grad()
    def forward(self, n_sweeps: int) -> Tuple[Tensor, Tensor]:
        '''Run the Metropolis-Hastings algorithm for a given number of sweeps'''
        #total_acceptance = 0.0 #debug
        for _ in range(n_sweeps):
            self.chains, self.log_prob, self.acceptance_rate = rw_metropolis_kernel_2D(logpdf=self.log_pdf,
                                                                                    position=self.chains,
                                                                                    log_prob=self.log_prob,
                                                                                    sigma=self.sigma)
            
            #total_acceptance += self.acceptance_rate.item() #(debug)
            if(self.target_acceptance is not None):
                #self.sigma = self.sigma
                self.sigma = self._update_sigma(self.acceptance_rate)
            else:
                self.sigma = self.sigma
        
        #avg_acceptance = 100 * total_acceptance / n_sweeps #debug
        #print(f"✅ Aceptación media tras {n_sweeps} sweeps: {avg_acceptance:.2f}%") #debug
        return self.chains, self.log_prob 



def rw_metropolis_kernel_Boundary(logpdf: Callable, position: Tensor, log_prob: Tensor, sigma: float, boundary: float = 6.0):
    """Random-Walk Metropolis-Hastings kernel with per-walker boundary constraints.
    
    Args:
        logpdf: Function that returns log probability
        position: Current positions [nwalkers, nfermions, dim]
        log_prob: Current log probabilities [nwalkers]
        sigma: Step size for proposals
        boundary: Size of boundary box (default: 6.0)
        dim: Number of spatial dimensions
    """
    # Generate proposals for all walkers
    proposal = position + sigma * torch.randn(position.shape, device=position.device)
    
    # Create boundary tensor matching the dimensionality
    boundaries = torch.full_like(proposal, boundary)
    
    # Check which individual proposals are within bounds
    # This will work for any number of dimensions in position.shape[-1]
    in_bounds = torch.all(torch.abs(proposal) <= boundaries, dim=(1,2))
    
    
    # Compute log probabilities for all proposals
    proposal_logprob = logpdf(proposal)
    
    # Standard Metropolis acceptance criterion
    log_uniform = torch.log(torch.rand(proposal_logprob.shape, device=position.device))
    accept = (log_uniform < proposal_logprob - log_prob)
    
    # Only accept moves that are both probabilistically accepted AND within bounds
    accept = accept & in_bounds
    
    # Update positions and log probabilities
    position = accept[:, None, None] * proposal + (~accept[:, None, None]) * position
    log_prob = torch.where(accept, proposal_logprob, log_prob)
    
    # Compute acceptance rate
    acceptance_rate = accept.float().mean()
    
    return position, log_prob, acceptance_rate



class MetropolisHastings_Boundary(nn.Module):

    def __init__(self, network: nn.Module, dof: int, nwalkers: int, target_acceptance: float, dim: int=1) -> None:
        super(MetropolisHastings_Boundary, self).__init__()

        # Set the parameters
        self.network = network #network to be trained
        self.dof = dof #number of fermions
        self.nwalkers = nwalkers #number of walkers
        self.target_acceptance = target_acceptance #target acceptance rate
        self.dim = dim #dimension of the system

        self.device = next(self.network.parameters()).device #device to run the simulation

        self.sigma = torch.tensor(1.0, device=self.device) #standard deviation of the proposal distribution 1.0
        self.acceptance_rate = torch.tensor(0.0, device=self.device) #acceptance rate
        #torch.manual_seed(42)
        self.chains = torch.randn(size=(self.nwalkers, self.dof, self.dim), #modified here 
                                        device=self.device,
                                        requires_grad=False) #isotropic initialisation of the points following the distribution.
        #print(f"Pos inicial {self.chains.squeeze(-1)}")
        _pretrain = self.network.pretrain
        if(_pretrain==True):
            self.network.pretrain=False  #inside the Metropolis we work with the original network
        self.log_prob = self.network(self.chains)[1].mul(2)
        self.network.pretrain=_pretrain

    def log_pdf(self, x: Tensor,alpha: float = 1.0) -> Tensor:
      '''2*Logarithm of the probability density function, and alpha reemphorces the curvature of the wavefunction'''
      _pretrain = self.network.pretrain
      if(_pretrain==True):
        self.network.pretrain=False
      _, logabs = self.network(x)
      self.network.pretrain=_pretrain
      return 2.*logabs*alpha

    def _update_sigma(self, acceptance_rate: Tensor) -> Tensor: 
        '''Update the standard deviation of the proposal distribution'''
        #Shamelessly taken from https://github.com/deepqmc/deepqmc
        scale_factor = self.target_acceptance / max(acceptance_rate, 0.05)
        return self.sigma / scale_factor
    
    def _initialize_walkers(self):
        """Inicializa los walkers desde una distribución normal. (Añadido para debugging)"""
        self.chains = torch.randn(size=(self.nwalkers, self.dof, self.dim),
                                  device=self.device,
                                  requires_grad=False)
        _pretrain = self.network.pretrain
        if _pretrain:
            self.network.pretrain = False
        self.log_prob = self.network(self.chains)[1].mul(2)
        self.network.pretrain = _pretrain

    def reset_walkers(self):
        """Reinicia los walkers para que empiecen desde cero. (Añadido para debugging)"""
        self._initialize_walkers()
        print("Walkers reseteados a su estado inicial.")

    @torch.no_grad()
    def forward(self, n_sweeps: int) -> Tuple[Tensor, Tensor]:
        '''Run the Metropolis-Hastings algorithm for a given number of sweeps'''
        #total_acceptance = 0.0 #debug
        for _ in range(n_sweeps):
            self.chains, self.log_prob, self.acceptance_rate = rw_metropolis_kernel_Boundary(logpdf=self.log_pdf,
                                                                                    position=self.chains,
                                                                                    log_prob=self.log_prob,
                                                                                    sigma=self.sigma)
            
            #total_acceptance += self.acceptance_rate.item() #(debug)
            if(self.target_acceptance is not None):
                #self.sigma = self.sigma
                self.sigma = self._update_sigma(self.acceptance_rate)
            else:
                self.sigma = self.sigma
        
        #avg_acceptance = 100 * total_acceptance / n_sweeps #debug
        #print(f"✅ Aceptación media tras {n_sweeps} sweeps: {avg_acceptance:.2f}%") #debug
        return self.chains, self.log_prob 


def rw_metropolis_kernel(pdf: Callable[[Tensor], Tensor], 
                             position: Tensor, 
                             prob: Tensor,  # <––– ESTE ES EL ARGUMENTO QUE FALTABA
                             sigma: float) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Random-Walk Metropolis-Hastings kernel en el dominio exponencial (sin logs).
    
    Args:
        pdf: función que devuelve directamente la densidad de probabilidad |ψ(x)|²
        position: tensor de posiciones actuales [nwalkers, dof, dim]
        prob: tensor con las probabilidades actuales [nwalkers]
        sigma: desviación estándar de la propuesta

    Returns:
        Nueva posición, nueva probabilidad, tasa de aceptación
    """
    #Shamelessly taken from https://rlouf.github.io/post/jax-random-walk-metropolis/
    
    proposal = position + sigma * torch.randn(position.shape, device=position.device)
    
    proposal_prob = pdf(proposal)
    
    ratio = proposal_prob / prob
    u = torch.rand_like(ratio)

    accept = u < ratio
    acceptance_rate = accept.float().mean()

    position = torch.where(accept[:, None, None], proposal, position)
    prob = torch.where(accept, proposal_prob, prob)

    return position, prob, acceptance_rate

class MetropolisHastings(nn.Module):

    def __init__(self, network: nn.Module, dof: int, nwalkers: int, target_acceptance: float, dim: int=1) -> None:
        super(MetropolisHastings, self).__init__()

        # Set the parameters
        self.network = network #network to be trained
        self.dof = dof #number of fermions
        self.nwalkers = nwalkers #number of walkers
        self.target_acceptance = target_acceptance #target acceptance rate
        self.dim = dim #dimension of the system

        self.device = next(self.network.parameters()).device #device to run the simulation

        self.sigma = torch.tensor(1.0, device=self.device) #standard deviation of the proposal distribution 1.0
        self.acceptance_rate = torch.tensor(0.0, device=self.device) #acceptance rate
        #torch.manual_seed(42)
        self.chains = torch.randn(size=(self.nwalkers, self.dof, self.dim), #modified here 
                                        device=self.device,
                                        requires_grad=False) #isotropic initialisation of the points following the distribution.
        #print(f"Pos inicial {self.chains.squeeze(-1)}")
        _pretrain = self.network.pretrain
        if(_pretrain==True):
            self.network.pretrain=False  #inside the Metropolis we work with the original network
        with torch.no_grad():
            _, logabs = self.network(self.chains)
            self.prob = torch.exp(2 * logabs)  # |ψ(x)|²
        self.network.pretrain=_pretrain

    def pdf(self, x: Tensor) -> Tensor:
        _pretrain = self.network.pretrain
        if _pretrain:
            self.network.pretrain = False
        _, logabs = self.network(x)
        self.network.pretrain = _pretrain
        return torch.exp(2 * logabs)

    def _update_sigma(self, acceptance_rate: Tensor) -> Tensor: 
        '''Update the standard deviation of the proposal distribution'''
        #Shamelessly taken from https://github.com/deepqmc/deepqmc
        scale_factor = self.target_acceptance / max(acceptance_rate, 0.05)
        return self.sigma / scale_factor
    
    def _initialize_walkers(self):
        """Inicializa los walkers desde una distribución normal. (Añadido para debugging)"""
        self.chains = torch.randn(size=(self.nwalkers, self.dof, self.dim),
                                  device=self.device,
                                  requires_grad=False)
        _pretrain = self.network.pretrain
        if _pretrain:
            self.network.pretrain = False
        self.log_prob = self.network(self.chains)[1].mul(2)
        self.network.pretrain = _pretrain

    def reset_walkers(self):
        """Reinicia los walkers para que empiecen desde cero. (Añadido para debugging)"""
        self._initialize_walkers()
        print("Walkers reseteados a su estado inicial.")

    @torch.no_grad()
    def forward(self, n_sweeps: int) -> Tuple[Tensor, Tensor]:
        '''Run the Metropolis-Hastings algorithm for a given number of sweeps'''
        #total_acceptance = 0.0 #debug
        for _ in range(n_sweeps):
            self.chains, self.log_prob, self.acceptance_rate =rw_metropolis_kernel(pdf=self.pdf,
                                                                                       position=self.chains,
                                                                                       prob=self.prob,
                                                                                       sigma=self.sigma
                                                                                       )
            
            #total_acceptance += self.acceptance_rate.item() #(debug)
            if(self.target_acceptance is not None):
                #self.sigma = self.sigma
                self.sigma = self._update_sigma(self.acceptance_rate)
            else:
                self.sigma = self.sigma
        
        #avg_acceptance = 100 * total_acceptance / n_sweeps #debug
        #print(f"✅ Aceptación media tras {n_sweeps} sweeps: {avg_acceptance:.2f}%") #debug
        return self.chains, self.log_prob 



def rw_metropolis_kernel_Old(logpdf: Callable, position: Tensor, log_prob: Tensor, sigma: float, dim: int=1):
    #Shamelessly taken from https://rlouf.github.io/post/jax-random-walk-metropolis/
    
    proposal = position + sigma * torch.randn(position.shape, device=position.device)
    
    proposal_logprob = logpdf(proposal)
    log_uniform = torch.log(torch.rand(proposal_logprob.shape, device=position.device))
    accept = log_uniform  < proposal_logprob - log_prob
    
    acceptance_rate = accept.float().mean()

    position = accept[:, None, None] * proposal + (~accept[:, None, None]) * position
    log_prob = torch.where(accept, proposal_logprob, log_prob)
    return position, log_prob, acceptance_rate



class MetropolisHastingsOld(nn.Module):

    def __init__(self, network: nn.Module, dof: int, nwalkers: int, target_acceptance: float, dim: int=1) -> None:
        super(MetropolisHastingsOld, self).__init__()

        # Set the parameters
        self.network = network #network to be trained
        self.dof = dof #number of fermions
        self.nwalkers = nwalkers #number of walkers
        self.target_acceptance = target_acceptance #target acceptance rate
        self.dim = dim #dimension of the system

        self.device = next(self.network.parameters()).device #device to run the simulation

        self.sigma = torch.tensor(1.0, device=self.device) #standard deviation of the proposal distribution 1.0
        self.acceptance_rate = torch.tensor(0.0, device=self.device) #acceptance rate
        #torch.manual_seed(42)
        self.chains = torch.randn(size=(self.nwalkers, self.dof, self.dim), #modified here 
                                        device=self.device,
                                        requires_grad=False) #isotropic initialisation of the points following the distribution.
        #print(f"Pos inicial {self.chains.squeeze(-1)}")
        _pretrain = self.network.pretrain
        if(_pretrain==True):
            self.network.pretrain=False  #inside the Metropolis we work with the original network
        self.log_prob = self.network(self.chains)[1].mul(2)
        self.network.pretrain=_pretrain

    def log_pdf(self, x: Tensor,alpha: float = 1.0) -> Tensor:
      '''2*Logarithm of the probability density function, and alpha reemphorces the curvature of the wavefunction'''
      _pretrain = self.network.pretrain
      if(_pretrain==True):
        self.network.pretrain=False
      _, logabs = self.network(x)
      self.network.pretrain=_pretrain
      return 2.*logabs*alpha

    def _update_sigma(self, acceptance_rate: Tensor) -> Tensor: 
        '''Update the standard deviation of the proposal distribution'''
        #Shamelessly taken from https://github.com/deepqmc/deepqmc
        scale_factor = self.target_acceptance / max(acceptance_rate, 0.05)
        return self.sigma / scale_factor
    
    def _initialize_walkers(self):
        """Inicializa los walkers desde una distribución normal. (Añadido para debugging)"""
        self.chains = torch.randn(size=(self.nwalkers, self.dof, self.dim),
                                  device=self.device,
                                  requires_grad=False)
        _pretrain = self.network.pretrain
        if _pretrain:
            self.network.pretrain = False
        self.log_prob = self.network(self.chains)[1].mul(2)
        self.network.pretrain = _pretrain

    def reset_walkers(self):
        """Reinicia los walkers para que empiecen desde cero. (Añadido para debugging)"""
        self._initialize_walkers()
        print("Walkers reseteados a su estado inicial.")

    @torch.no_grad()
    def forward(self, n_sweeps: int) -> Tuple[Tensor, Tensor]:
        '''Run the Metropolis-Hastings algorithm for a given number of sweeps'''
        #total_acceptance = 0.0 #debug
        for _ in range(n_sweeps):
            self.chains, self.log_prob, self.acceptance_rate = rw_metropolis_kernel_Old(logpdf=self.log_pdf,
                                                                                    position=self.chains,
                                                                                    log_prob=self.log_prob,
                                                                                    sigma=self.sigma)
            
            #total_acceptance += self.acceptance_rate.item() #(debug)
            if(self.target_acceptance is not None):
                #self.sigma = self.sigma
                self.sigma = self._update_sigma(self.acceptance_rate)
            else:
                self.sigma = self.sigma
        
        #avg_acceptance = 100 * total_acceptance / n_sweeps #debug
        #print(f"✅ Aceptación media tras {n_sweeps} sweeps: {avg_acceptance:.2f}%") #debug
        return self.chains, self.log_prob 


####################################################################################################################################################
############################################################## SAMPLER 2.0 #####################################################################################################
####################################################################################################################################################

# -------------  helpers -----------------------------------------------------
def _reflect_in_ball(x: torch.Tensor, L: float) -> torch.Tensor:
    """Radially reflect points that lie outside the D-sphere of radius L."""
    v = x.view(x.shape[0], -1)                       # [nwalkers, D]
    r = torch.linalg.vector_norm(v, dim=1)           # [nwalkers]
    mask = r > L
    if mask.any():
        v_out = v[mask]
        v_ref = (L / r[mask])[:, None] * v_out       # scale onto sphere
        x[mask] = v_ref.view_as(x[mask])             # back into original shape
    return x
# ----------------------------------------------------------------------------


def mala_kernel_reflect(
        logpdf: Callable,                 # returns log πθ per walker
        position: torch.Tensor,           # [nwalkers, dof, dim]
        log_prob: torch.Tensor,           # [nwalkers]
        sigma: float,
        L: float = 6.0
    ):
    """
    Metropolis-adjusted Langevin proposal + radial reflection.
    Detailed balance w.r.t. the *original* density is preserved.
    """
    # ------------------------------------------------------------------ step 1
    # Compute ∇log π at the current positions
    position.requires_grad_(True)
    logp_x = logpdf(position)            # [nwalkers]
    grad_x = torch.autograd.grad(logp_x.sum(), position)[0]
    position = position.detach()         # stop autograd graph

    # Langevin proposal
    eps = sigma
    noise = torch.randn_like(position)
    drift = 0.5 * eps**2 * grad_x
    proposal = position + drift + eps * noise

    # ------------------------------------------------------------------ step 2
    # Reflect if the proposal left the safe sphere
    #proposal = _reflect_in_ball(proposal, L=L)

    # ------------------------------------------------------------------ step 3
    # Compute log π and ∇log π at the proposal (needed for Hastings term)
    proposal.requires_grad_(True)
    logp_y = logpdf(proposal)
    grad_y = torch.autograd.grad(logp_y.sum(), proposal)[0]
    proposal = proposal.detach()

    # Forward / backward Gaussian densities (log-space)
    def _log_q(x, y, grad_x):
        mean = x + 0.5 * eps**2 * grad_x
        diff = (y - mean).view(y.shape[0], -1)
        return -0.5 / eps**2 * (diff ** 2).sum(dim=1)

    log_q_xy = _log_q(position, proposal, grad_x)
    log_q_yx = _log_q(proposal, position, grad_y)

    # Metropolis–Hastings acceptance
    log_alpha = logp_y - logp_x + log_q_yx - log_q_xy
    accept = (torch.log(torch.rand_like(log_alpha)) < log_alpha)

    # ------------------------------------------------------------------ step 4
    new_pos = torch.where(accept[:, None, None], proposal, position)
    new_logp = torch.where(accept, logp_y, logp_x)
    acc_rate = accept.float().mean()

    return new_pos, new_logp, acc_rate

class MetropolisHastings_2(nn.Module):
    def __init__(self, network, dof, nwalkers,
                 target_acceptance=0.5, dim=1, L=6.0):
        super().__init__()
        self.network, self.dof, self.dim = network, dof, dim
        self.nwalkers, self.target_acceptance, self.L = nwalkers, target_acceptance, L
        self.device = next(network.parameters()).device

        # ---- scaled initial σ (Roberts–Gelman–Gilks) ------------------
        D = dof * dim
        self.sigma = torch.tensor(2.4 / math.sqrt(D), device=self.device)
        # ---------------------------------------------------------------

        self._initialize_walkers()

    # unchanged utility functions (log_pdf, _update_sigma, _init_walkers, …)
    def log_pdf(self, x: Tensor,alpha: float = 1.0) -> Tensor:
      '''2*Logarithm of the probability density function, and alpha reemphorces the curvature of the wavefunction'''
      _pretrain = self.network.pretrain
      if(_pretrain==True):
        self.network.pretrain=False
      _, logabs = self.network(x)
      self.network.pretrain=_pretrain
      return 2.*logabs*alpha

    def _update_sigma(self, acceptance_rate: Tensor) -> Tensor: 
        '''Update the standard deviation of the proposal distribution'''
        #Shamelessly taken from https://github.com/deepqmc/deepqmc
        scale_factor = self.target_acceptance / max(acceptance_rate, 0.05)
        return self.sigma / scale_factor
    
    def _initialize_walkers(self):
        """Inicializa los walkers desde una distribución normal. (Añadido para debugging)"""
        self.chains = torch.randn(size=(self.nwalkers, self.dof, self.dim),
                                  device=self.device,
                                  requires_grad=False)
        _pretrain = self.network.pretrain
        if _pretrain:
            self.network.pretrain = False
        self.log_prob = self.network(self.chains)[1].mul(2)
        self.network.pretrain = _pretrain

    def reset_walkers(self):
        """Reinicia los walkers para que empiecen desde cero. (Añadido para debugging)"""
        self._initialize_walkers()
        print("Walkers reseteados a su estado inicial.")


    #@torch.no_grad()
    def forward(self, n_sweeps: int):
        for _ in range(n_sweeps):
            self.chains, self.log_prob, self.acceptance_rate = (
                mala_kernel_reflect(
                    self.log_pdf, self.chains, self.log_prob,
                    sigma=float(self.sigma), L=self.L)
            )
            with torch.no_grad():
                if self.target_acceptance is not None:
                    self.sigma = self._update_sigma(self.acceptance_rate)
        return self.chains, self.log_prob


###########################################################################################################################################################
############################################################# Adaptative sigma #####################################################################
############################################################################################################################################################
def rw_metropolis_kernel_sigma(logpdf: Callable, position: Tensor, log_prob: Tensor, sigma: float, dim: int=1):
    #Shamelessly taken from https://rlouf.github.io/post/jax-random-walk-metropolis/
    
    proposal = position + sigma * torch.randn(position.shape, device=position.device)
    
    proposal_logprob = logpdf(proposal)
    log_uniform = torch.log(torch.rand(proposal_logprob.shape, device=position.device))
    accept = log_uniform  < proposal_logprob - log_prob
    
    acceptance_rate = accept.float().mean()

    position = accept[:, None, None] * proposal + (~accept[:, None, None]) * position
    log_prob = torch.where(accept, proposal_logprob, log_prob)
    return position, log_prob, acceptance_rate



class MetropolisHastings_sigma(nn.Module):

    def __init__(self, network: nn.Module, dof: int, nwalkers: int, target_acceptance: float, dim: int=1) -> None:
        super(MetropolisHastings_sigma, self).__init__()

        # Set the parameters
        self.network = network #network to be trained
        self.dof = dof #number of fermions
        self.nwalkers = nwalkers #number of walkers
        self.target_acceptance = target_acceptance #target acceptance rate
        self.dim = dim #dimension of the system

        self.device = next(self.network.parameters()).device #device to run the simulation

        D = self.dof * self.dim          # total number of coordinates
        self.sigma = torch.tensor(2.4 / math.sqrt(D), device=self.device)
        self.acceptance_rate = torch.tensor(0.0, device=self.device) #acceptance rate
        #torch.manual_seed(42)
        self.chains = torch.randn(size=(self.nwalkers, self.dof, self.dim), #modified here 
                                        device=self.device,
                                        requires_grad=False) #isotropic initialisation of the points following the distribution.
        #print(f"Pos inicial {self.chains.squeeze(-1)}")
        _pretrain = self.network.pretrain
        if(_pretrain==True):
            self.network.pretrain=False  #inside the Metropolis we work with the original network
        self.log_prob = self.network(self.chains)[1].mul(2)
        self.network.pretrain=_pretrain

    def log_pdf(self, x: Tensor,alpha: float = 1.0) -> Tensor:
      '''2*Logarithm of the probability density function, and alpha reemphorces the curvature of the wavefunction'''
      _pretrain = self.network.pretrain
      if(_pretrain==True):
        self.network.pretrain=False
      _, logabs = self.network(x)
      self.network.pretrain=_pretrain
      return 2.*logabs*alpha

    def _update_sigma(self, acceptance_rate: Tensor) -> Tensor: 
        '''Update the standard deviation of the proposal distribution'''
        eps = 1e-4
        acc_clamped = torch.clamp(acceptance_rate, eps, 1.0)        # [eps, 1]
        ratio = acc_clamped / self.target_acceptance
        sigma_new = self.sigma / ratio

        # límites duros para σ
        D = self.dof * self.dim
        σ_min, σ_max = 0.05 / math.sqrt(D), 5.0 / math.sqrt(D)
        sigma_new = torch.clamp(sigma_new, σ_min, σ_max)

        # si algo salió mal, deja la σ anterior
        if torch.isnan(sigma_new) or torch.isinf(sigma_new):
            sigma_new = self.sigma.clone()

        return sigma_new
    
    def _initialize_walkers(self):
        """Inicializa los walkers desde una distribución normal. (Añadido para debugging)"""
        self.chains = torch.randn(size=(self.nwalkers, self.dof, self.dim),
                                  device=self.device,
                                  requires_grad=False)
        _pretrain = self.network.pretrain
        if _pretrain:
            self.network.pretrain = False
        self.log_prob = self.network(self.chains)[1].mul(2)
        self.network.pretrain = _pretrain

    def reset_walkers(self):
        """Reinicia los walkers para que empiecen desde cero. (Añadido para debugging)"""
        self._initialize_walkers()
        print("Walkers reseteados a su estado inicial.")

    @torch.no_grad()
    def forward(self, n_sweeps: int) -> Tuple[Tensor, Tensor]:
        '''Run the Metropolis-Hastings algorithm for a given number of sweeps'''
        #total_acceptance = 0.0 #debug
        for _ in range(n_sweeps):
            self.chains, self.log_prob, self.acceptance_rate = rw_metropolis_kernel_Old(logpdf=self.log_pdf,
                                                                                    position=self.chains,
                                                                                    log_prob=self.log_prob,
                                                                                    sigma=self.sigma)
            
            #total_acceptance += self.acceptance_rate.item() #(debug)
            if(self.target_acceptance is not None):
                #self.sigma = self.sigma
                self.sigma = self._update_sigma(self.acceptance_rate)
            else:
                self.sigma = self.sigma
        
        #avg_acceptance = 100 * total_acceptance / n_sweeps #debug
        #print(f"✅ Aceptación media tras {n_sweeps} sweeps: {avg_acceptance:.2f}%") #debug
        return self.chains, self.log_prob 


##################################################################################################################################################
################################################################# Log-envelope ###################################################################
##################################################################################################################################################




class MetropolisHastings_envelope(nn.Module):

    def __init__(self, network: nn.Module, dof: int, nwalkers: int, target_acceptance: float, dim: int=1) -> None:
        super(MetropolisHastings_envelope, self).__init__()

        # Set the parameters
        self.network = network #network to be trained
        self.dof = dof #number of fermions
        self.nwalkers = nwalkers #number of walkers
        self.target_acceptance = target_acceptance #target acceptance rate
        self.dim = dim #dimension of the system

        self.device = next(self.network.parameters()).device #device to run the simulation

        self.sigma = torch.tensor(1.0, device=self.device) #standard deviation of the proposal distribution 1.0
        self.acceptance_rate = torch.tensor(0.0, device=self.device) #acceptance rate
        #torch.manual_seed(42)
        self.chains = torch.randn(size=(self.nwalkers, self.dof, self.dim), #modified here 
                                        device=self.device,
                                        requires_grad=False) #isotropic initialisation of the points following the distribution.
        #print(f"Pos inicial {self.chains.squeeze(-1)}")
        _pretrain = self.network.pretrain
        if(_pretrain==True):
            self.network.pretrain=False  #inside the Metropolis we work with the original network
        self.log_prob = self.network(self.chains)[1].mul(2)
        self.network.pretrain=_pretrain

    def log_pdf(self, x: Tensor,alpha: float = 1.0,envelope_strength: float = 1e-1) -> Tensor:
        """2*Logarithm of the probability density function, optionally with a Gaussian envelope penalty."""
        _pretrain = self.network.pretrain
        if _pretrain:
            self.network.pretrain = False
        _, logabs = self.network(x)
        self.network.pretrain = _pretrain

        # Gaussian envelope: -lambda * ||x||^2
        # Penalizes walkers from straying too far from the origin
        envelope = -envelope_strength * (x ** 2).sum(dim=(1, 2))

        return 2. * logabs * alpha + envelope

    def _update_sigma(self, acceptance_rate: Tensor) -> Tensor: 
        '''Update the standard deviation of the proposal distribution'''
        #Shamelessly taken from https://github.com/deepqmc/deepqmc
        scale_factor = self.target_acceptance / max(acceptance_rate, 0.05)
        return self.sigma / scale_factor
    
    def _initialize_walkers(self):
        """Inicializa los walkers desde una distribución normal. (Añadido para debugging)"""
        self.chains = torch.randn(size=(self.nwalkers, self.dof, self.dim),
                                  device=self.device,
                                  requires_grad=False)
        _pretrain = self.network.pretrain
        if _pretrain:
            self.network.pretrain = False
        self.log_prob = self.network(self.chains)[1].mul(2)
        self.network.pretrain = _pretrain

    def reset_walkers(self):
        """Reinicia los walkers para que empiecen desde cero. (Añadido para debugging)"""
        self._initialize_walkers()
        print("Walkers reseteados a su estado inicial.")

    @torch.no_grad()
    def forward(self, n_sweeps: int) -> Tuple[Tensor, Tensor]:
        '''Run the Metropolis-Hastings algorithm for a given number of sweeps'''
        #total_acceptance = 0.0 #debug
        for _ in range(n_sweeps):
            self.chains, self.log_prob, self.acceptance_rate = rw_metropolis_kernel_Old(logpdf=self.log_pdf,
                                                                                    position=self.chains,
                                                                                    log_prob=self.log_prob,
                                                                                    sigma=self.sigma)
            
            #total_acceptance += self.acceptance_rate.item() #(debug)
            if(self.target_acceptance is not None):
                #self.sigma = self.sigma
                self.sigma = self._update_sigma(self.acceptance_rate)
            else:
                self.sigma = self.sigma
        
        #avg_acceptance = 100 * total_acceptance / n_sweeps #debug
        #print(f"✅ Aceptación media tras {n_sweeps} sweeps: {avg_acceptance:.2f}%") #debug
        return self.chains, self.log_prob 


