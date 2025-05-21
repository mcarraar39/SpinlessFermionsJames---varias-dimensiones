#Pytorch package
import torch
from torch import nn, Tensor

#Typecasting
from typing import Tuple, Callable

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
    accept = log_uniform < proposal_logprob - log_prob
    
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