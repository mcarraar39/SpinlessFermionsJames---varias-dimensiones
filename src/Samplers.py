#Pytorch package
import torch
from torch import nn, Tensor

#Typecasting
from typing import Tuple, Callable

def rw_metropolis_kernel(logpdf: Callable, position: Tensor, log_prob: Tensor, sigma: float, dim: int=1):
    #Shamelessly taken from https://rlouf.github.io/post/jax-random-walk-metropolis/
    
    proposal = position + sigma * torch.randn(position.shape, device=position.device)
    
    proposal_logprob = logpdf(proposal)
    log_uniform = torch.log(torch.rand(proposal_logprob.shape, device=position.device))
    accept = log_uniform < proposal_logprob - log_prob
    
    acceptance_rate = accept.float().mean()

    position = accept[:, None, None] * proposal + (~accept[:, None, None]) * position
    log_prob = torch.where(accept, proposal_logprob, log_prob)
    return position, log_prob, acceptance_rate

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

        self.sigma = torch.tensor(1.0, device=self.device) #standard deviation of the proposal distribution
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

    def log_pdf(self, x: Tensor) -> Tensor:
      '''2*Logarithm of the probability density function'''
      _pretrain = self.network.pretrain
      if(_pretrain==True):
        self.network.pretrain=False
      _, logabs = self.network(x)
      self.network.pretrain=_pretrain
      return 2.*logabs

    def _update_sigma(self, acceptance_rate: Tensor) -> Tensor: 
        '''Update the standard deviation of the proposal distribution'''
        #Shamelessly taken from https://github.com/deepqmc/deepqmc
        scale_factor = self.target_acceptance / max(acceptance_rate, 0.05)
        return self.sigma / scale_factor
    
    def _initialize_walkers(self):
        """Inicializa los walkers desde una distribución normal."""
        self.chains = torch.randn(size=(self.nwalkers, self.dof, self.dim),
                                  device=self.device,
                                  requires_grad=False)
        _pretrain = self.network.pretrain
        if _pretrain:
            self.network.pretrain = False
        self.log_prob = self.network(self.chains)[1].mul(2)
        self.network.pretrain = _pretrain

    def reset_walkers(self):
        """Reinicia los walkers para que empiecen desde cero."""
        self._initialize_walkers()
        print("Walkers reseteados a su estado inicial.")

    @torch.no_grad()
    def forward(self, n_sweeps: int) -> Tuple[Tensor, Tensor]:
        '''Run the Metropolis-Hastings algorithm for a given number of sweeps'''
        for _ in range(n_sweeps):
            self.chains, self.log_prob, self.acceptance_rate = rw_metropolis_kernel(logpdf=self.log_pdf,
                                                                                    position=self.chains,
                                                                                    log_prob=self.log_prob,
                                                                                    sigma=self.sigma)
            
            if(self.target_acceptance is not None):
              self.sigma = self._update_sigma(self.acceptance_rate)
            else:
              self.sigma = self.sigma
        return self.chains, self.log_prob 