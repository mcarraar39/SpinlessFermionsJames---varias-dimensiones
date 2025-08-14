import torch
import torch.nn as nn
import math, sys

from torch import Tensor
from typing import Tuple

#from Functions import GeneralisedLogSumExpEnvLogDomainStable
#from Functions import NaiveLogSumExpEnvLogDomainStable
from Functions import custom_function, naive_function

class EquivariantLayer0(nn.Module):

  def __init__(self, in_features: int, out_features: int, num_particles: int, bias: bool) -> None:
    """Equivariant layer which takes in a batch of matrices and returns a batch of matrices 
    representing the hidden features in a permutationally equivariant manner. The number 
    of output features must be greater than the number of number of particles,
    `out_features` > `num_particles` to ensure that the network's output is non-singular.
    
    :param in_features: number of input features for the equivariant layer
    :type in_features: int
    
    :param out_features: number of output features for the equivariant layer
    :type out_features: int
    
    :param num_particles: number of particles for the network
    :type num_particles: int
       
    :param bias: A flag to determine if the `nn.Linear` object uses a bias
    :type bias: bool
    
    :return out: None
    :type out: None
    """
    super(EquivariantLayer0, self).__init__()
    
    self.in_features = in_features
    self.out_features = out_features
    self.num_particles = num_particles
    self.bias = bias
    
    self.fc = nn.Linear(self.in_features, self.out_features, bias=self.bias)

    
  def forward(self, h: Tensor) -> Tensor:
    """
    h: Tensor of shape [nwalkers, A, D]
    Returns: Tensor of shape [nwalkers, A, out_features] with appended mean feature
    """
    # Compute mean per walker (scalar): [nwalkers]
    walker_mean = h.mean(dim=(1, 2), keepdim=True)  # shape: [nwalkers, 1, 1]

    # Expand to match particle shape: [nwalkers, A, 1]
    walker_mean_expanded = walker_mean.expand(-1, self.num_particles, 1)

    # Concatenate mean to each particle's feature vector
    f = torch.cat((h, walker_mean_expanded), dim=-1)  # shape: [nwalkers, A, D+1]

    # Apply linear layer
    return self.fc(f)  # shape: [nwalkers, A, out_features]

class EquivariantLayer1(nn.Module):

  def __init__(self, in_features: int, out_features: int, num_particles: int, bias: bool) -> None:
    """Equivariant layer which takes in a batch of matrices and returns a batch of matrices 
    representing the hidden features in a permutationally equivariant manner. The number 
    of output features must be greater than the number of number of particles,
    `out_features` > `num_particles` to ensure that the network's output is non-singular.
    
    :param in_features: number of input features for the equivariant layer
    :type in_features: int
    
    :param out_features: number of output features for the equivariant layer
    :type out_features: int
    
    :param num_particles: number of particles for the network
    :type num_particles: int
       
    :param bias: A flag to determine if the `nn.Linear` object uses a bias
    :type bias: bool
    
    :return out: None
    :type out: None

    The mean here is more general
    """
    super(EquivariantLayer1, self).__init__()
    
    self.in_features = in_features
    self.out_features = out_features
    self.num_particles = num_particles
    self.bias = bias
    
    self.fc = nn.Linear(self.in_features, self.out_features, bias=self.bias)

    
  def forward(self, h: Tensor) -> Tensor:
    """The call method of the layer 
    :param h: Input tensor of shape [nwalkers, A, D] where:
             - A: number of particles
             - D: number of features per particle
    :return out: Output tensor of shape [nwalkers, A, out_features]
    """
    # Average over both particles (dim=-2) and features (dim=-1)
    g = h.mean(dim=(-2, -1), keepdim=True)  # Shape: [nwalkers, 1, 1]
    
    # Expand mean to match particle dimension
    rep = [1] * len(h.shape)
    rep[-2] = self.num_particles
    rep[-1] = 1
    g = g.repeat(*rep)  # Shape: [nwalkers, A, 1]
    
    # Concatenate original input and mean
    f = torch.cat((h, g), dim=-1)  # Shape: [nwalkers, A, D+1]
    #print('f0',f.shape)
    return self.fc(f)  # Shape: [nwalkers, A, out_features]

class EquivariantLayer2(nn.Module):

  def __init__(self, in_features: int, out_features: int, num_particles: int, bias: bool) -> None:
    """Equivariant layer which takes in a batch of matrices and returns a batch of matrices 
    representing the hidden features in a permutationally equivariant manner. The number 
    of output features must be greater than the number of number of particles,
    `out_features` > `num_particles` to ensure that the network's output is non-singular.
    
    :param in_features: number of input features for the equivariant layer
    :type in_features: int
    
    :param out_features: number of output features for the equivariant layer
    :type out_features: int
    
    :param num_particles: number of particles for the network
    :type num_particles: int
       
    :param bias: A flag to determine if the `nn.Linear` object uses a bias
    :type bias: bool
    
    :return out: None
    :type out: None

    The mean here is adapted to be equal to the unidimensional case
    """
    super(EquivariantLayer2, self).__init__()
    
    self.in_features = in_features
    self.out_features = out_features
    self.num_particles = num_particles
    self.bias = bias
    
    self.fc = nn.Linear(self.in_features , self.out_features, bias=self.bias)


    
  def forward(self, h: Tensor) -> Tensor:
        # h: [nwalkers, A, D]
        g = h.mean(dim=-2, keepdim=True)  # [nwalkers, 1, D]
        g = g.repeat(1, self.num_particles, 1)  # [nwalkers, A, D]
        f = torch.cat((h, g), dim=-1)  # [nwalkers, A, 2D]
        return self.fc(f)  # [nwalkers, A, out_features]

class EquivariantLayer3(nn.Module):

  def __init__(self, in_features: int, out_features: int, num_particles: int, bias: bool) -> None:
    """Equivariant layer which takes in a batch of matrices and returns a batch of matrices 
    representing the hidden features in a permutationally equivariant manner. The number 
    of output features must be greater than the number of number of particles,
    `out_features` > `num_particles` to ensure that the network's output is non-singular.
    
    :param in_features: number of input features for the equivariant layer
    :type in_features: int
    
    :param out_features: number of output features for the equivariant layer
    :type out_features: int
    
    :param num_particles: number of particles for the network
    :type num_particles: int
       
    :param bias: A flag to determine if the `nn.Linear` object uses a bias
    :type bias: bool
    
    :return out: None
    :type out: None

    The mean here is more general
    """
    super(EquivariantLayer3, self).__init__()
    
    self.in_features = in_features
    self.out_features = out_features
    self.num_particles = num_particles
    self.bias = bias
    
    # Linear layer expecting D+1 input features
    self.fc = nn.Linear(self.in_features, self.out_features, bias=self.bias)

  def forward(self, h: Tensor) -> Tensor:
    """
    h: Input tensor of shape [nwalkers, A, D]
    Returns: Output tensor of shape [nwalkers, A, out_features]
    """
    # Average over particles only (preserve dimensions)
    g = h.mean(dim=-2, keepdim=True)  # Shape: [nwalkers, 1, D]
    
    # Expand mean to match particle dimension
    g = g.repeat(1, self.num_particles, 1)  # Shape: [nwalkers, A, D]
    
    # Concatenate original input and mean
    f = torch.cat((h, g), dim=-1)  # Shape: [nwalkers, A, D+1]
    #print('f0',f.shape)
    return self.fc(f)  # Shape: [nwalkers, A, out_features] [nwalkers, A, out_features]

class EquivariantLayer4(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_particles: int, bias: bool) -> None:
        """
        Equivariant layer which takes in a batch of matrices and returns a batch of matrices 
        representing the hidden features in a permutationally equivariant manner. The number 
        of output features must be greater than the number of number of particles,
        `out_features` > `num_particles` to ensure that the network's output is non-singular.
        
        :param in_features: number of input features for the equivariant layer
        :type in_features: int
        
        :param out_features: number of output features for the equivariant layer
        :type out_features: int
        
        :param num_particles: number of particles for the network
        :type num_particles: int
        
        :param bias: A flag to determine if the `nn.Linear` object uses a bias
        :type bias: bool
        """
        super(EquivariantLayer4, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_particles = num_particles
        self.bias = bias
        
        # Ajuste del tamaño de entrada al concatenar D promedios
        self.fc = nn.Linear(self.in_features, self.out_features, bias=self.bias)

    def forward(self, h: Tensor) -> Tensor:
        """
        h: Input tensor of shape [nwalkers, A, D] where:
           - A: number of particles
           - D: number of features per particle
        Returns: Output tensor of shape [nwalkers, A, out_features]
        """
        # Promedio por coordenada (dim=-2), preservando la dimensión
        g = h.mean(dim=-2, keepdim=True)  # Shape: [nwalkers, 1, D]
        
        # Expandir el promedio para cada partícula
        g = g.repeat(1, self.num_particles, 1)  # Shape: [nwalkers, A, D]
        #print(g.shape)
        # Concatenar el tensor original con los promedios
        f = torch.cat((h, g), dim=-1)  # Shape: [nwalkers, A, 2D]
        #print(f)
        # Aplicar la capa lineal
        return self.fc(f)  # Shape: [nwalkers, A, out_features]

  
class EquivariantLayer(nn.Module):

  def __init__(self, in_features: int, out_features: int, num_particles: int, bias: bool) -> None:
    """Equivariant layer which takes in a batch of matrices and returns a batch of matrices 
    representing the hidden features in a permutationally equivariant manner. The number 
    of output features must be greater than the number of number of particles,
    `out_features` > `num_particles` to ensure that the network's output is non-singular.
    
    :param in_features: number of input features for the equivariant layer
    :type in_features: int
    
    :param out_features: number of output features for the equivariant layer
    :type out_features: int
    
    :param num_particles: number of particles for the network
    :type num_particles: int
       
    :param bias: A flag to determine if the `nn.Linear` object uses a bias
    :type bias: bool
    
    :return out: None
    :type out: None
    """
    super(EquivariantLayer, self).__init__()
    
    self.in_features = in_features
    self.out_features = out_features
    self.num_particles = num_particles
    self.bias = bias
    
    self.fc = nn.Linear(self.in_features, self.out_features, bias=self.bias)

    
  def forward(self, h: Tensor) -> Tensor:
    """The call method of the layer 
    :param h: Input tensor containing the propagated features from the previous layer
    :type h: class: `torch.Tensor`
        
    :return out: Output tensor containing the output features from the current layer
                 after being pass through `torch.nn.Linear` and the layer
                 corresponding non-linear activation function.
    :rtype out: class: `torch.Tensor`
    """
    idx=len(h.shape)
    rep=[1 for _ in range(idx)]
    rep[-2] = self.num_particles
    
    g = h.mean(dim=(idx-2), keepdim=True).repeat(*rep)
    f = torch.cat((h,g), dim=-1)
    #print('f2',f.shape)
    return self.fc(f)

class SlaterMultiDet(nn.Module):
  
  def __init__(self, in_features, num_particles, num_dets, bias) -> None:
    """The Multi Slater Determinant layer which takes in a feature input Tensor of shape [, A, H]
       and performs a multi-linear dot-product to create a set of D Generalised Slater Matrices
       (and with bias if bias=True). The input feature Tensor must have H > A to ensure that the 
       Generalised Slater Matrices are non-singular. 
    
    :param in_features: number of input features for the equivariant layer
    :type in_features: int
    
    :param num_particles: number of particles for the network
    :type num_particles: int
    
    :param num_dets: number of Generalised Slater Matrices 
    :type num_dets: int
    
    :param bias: A flag to determine if the `nn.Linear` object uses a bias
    :type bias: bool
    """
    super(SlaterMultiDet, self).__init__()
    
    self.in_features = in_features
    self.num_particles = num_particles
    self.num_dets = num_dets
    self.bias = bias

    self.gsds = nn.ModuleList([nn.Linear(self.in_features, self.num_particles, bias=self.bias) for _ in range(self.num_dets)])
    
  def forward(self, h: Tensor) -> Tensor:
    """Call method of the class
       
       :param h: Input Tensor containing the features of the last equivariant layer
       :type h: class: `torch.Tensor`
       
       :return out: The D Generalised Slater Matrices (without Log-Envelopes)
       :rtype out: class: `torch.Tensor`
    """
    return torch.stack([layer(h) for layer in self.gsds], dim=-3)
  
class LogEnvelope(nn.Module):

  def __init__(self, num_particles: int, num_dets: int, Dim:int,bias: bool=False) -> None:
    """The Log-Envelope layer of the network. This layer takes in the many-body 
       particles positions and create a set of D matrices representing a gaussian envelope 
       for each orbital. 
       
       :param num_particles: number of particles within the network
       :type num_particles: int
       
       :param num_dets: number of Generalised Slater Matrices
       :type num_dets: int

       :param dim: The dimension of the system
        :type dim: int
       
       :param bias: A flag to determine if a bias is used. Default: False.
       :type bias: bool
    """
    super(LogEnvelope, self).__init__()
    
    self.num_particles = num_particles
    self.num_dets = num_dets
    self.bias = bias
    self.dim = Dim

    self.log_envs = nn.ModuleList([nn.Linear(self.dim, self.num_particles, bias=self.bias) for _ in range(self.num_dets)]) #cambio aqui
    
    self.reset_parameters()

  @torch.no_grad()
  def reset_parameters(self) -> None:
    """Initialises the initial parameters values for the `weight` and `bias` class attributes
       both are initialised via `torch.nn.init.uniform_`.    
    """
    for layer in self.log_envs:
      layer.weight.fill_(0.5**(0.5)) #speed up convergence. 
      

  def forward(self, x0: Tensor) -> Tensor: #modificado
    
    """
    Forward pass for LogEnvelope with multidimensional input (A, D).
    
    :param x0: Input positions of shape [A, D]
    :return: Output tensor of shape [num_dets, A, num_particles]
    """
    
    envs = torch.stack([layer(x0).pow(2).sum(dim=-1, keepdim=True) for layer in self.log_envs], dim=-3)
    # for i, layer in enumerate(self.log_envs):
    #   print(f"Weights for determinant {i}:")
    #   print(layer.weight)  # Shape: [num_particles, dim]
    return envs.mul(-1)

class LogEnvelopeOld(nn.Module):

  def __init__(self, num_particles: int, num_dets: int, Dim:int,bias: bool=False) -> None:
    """The Log-Envelope layer of the network. This layer takes in the many-body 
       particles positions and create a set of D matrices representing a gaussian envelope 
       for each orbital. 
       
       :param num_particles: number of particles within the network
       :type num_particles: int
       
       :param num_dets: number of Generalised Slater Matrices
       :type num_dets: int

       :param dim: The dimension of the system
        :type dim: int
       
       :param bias: A flag to determine if a bias is used. Default: False.
       :type bias: bool
    """
    super(LogEnvelope, self).__init__()
    
    self.num_particles = num_particles
    self.num_dets = num_dets
    self.bias = bias
    self.dim = Dim

    self.log_envs = nn.ModuleList([nn.Linear(self.dim, self.num_particles, bias=self.bias) for _ in range(self.num_dets)]) #cambio aqui
    
    self.reset_parameters()

  @torch.no_grad()
  def reset_parameters(self) -> None:
    """Initialises the initial parameters values for the `weight` and `bias` class attributes
       both are initialised via `torch.nn.init.uniform_`.    
    """
    for layer in self.log_envs:
      layer.weight.fill_(0.5**(0.5)) #speed up convergence. 
      

  def forward(self, x0: Tensor) -> Tensor: #modificado
    
    """
    Forward pass for LogEnvelope with multidimensional input (A, D).
    
    :param x0: Input positions of shape [A, D]
    :return: Output tensor of shape [num_dets, A, num_particles]
    """
    x = x0
    return torch.stack([layer(x) for layer in self.log_envs], dim=-3).pow(2).mul(-1)
    
class MatrixToSLogDeterminant(nn.Module):
  
  def __init__(self, num_particles: int) -> None:
    """The Multi Matrix to Signed Log Determinant Function takes in a set of 
       D Generalised Slater Matrices and returns a global logabs value (and global sign)
       via a custom `torch.autograd.Function` with numerically stable `Backward` and 
       `DoubleBackward` methods. 
       
       :param num_particles: the number of particles in the input layer of the network (assigned automatically)
       :type num_particles: int
       
    """
    super(MatrixToSLogDeterminant, self).__init__()
    self.num_particles = num_particles
    self.log_factorial = torch.arange(1,self.num_particles+1).float().log().sum() #normalisation factor of the determinant (within log-domain)

   
  def forward(self, matrices: Tensor, log_envs: Tensor) -> Tuple[Tensor, Tensor]:
    """Call method of the class
    
      :param matrices: Single Generalised Slater Matrices (with Log-Envelopes)
      :type matrices: class: `torch.Tensor`
      
      :return out: the global sign and global logabs values of the signed
                   log-determinant values of the D Generalised Slater Matrices
    """
    #sgn, logabs, _, _ = GeneralisedLogSumExpEnvLogDomainStable.apply(matrices, log_envs)
    ##sgn, logabs = NaiveLogSumExpEnvLogDomainStable(matrices, log_envs)
    sgn, logabs = custom_function.apply(matrices, log_envs) #replace with naive_function for standard pytorch primitives
    return sgn, logabs - 0.5*self.log_factorial
  
class LinearLayer(nn.Module):

  def __init__(self, in_features: int, out_features: int, num_particles: int, bias: bool) -> None:
    """Linear layer which takes in a batch of matrices and returns a batch of matrices 
    representing the hidden features in a permutationally equivariant manner. The number 
    of output features must be greater than the number of number of particles,
    `out_features` > `num_particles` to ensure that the network's output is non-singular.
    
    :param in_features: number of input features for the equivariant layer
    :type in_features: int
    
    :param out_features: number of output features for the equivariant layer
    :type out_features: int
    
    :param num_particles: number of particles for the network
    :type num_particles: int
       
    :param bias: A flag to determine if the `nn.Linear` object uses a bias
    :type bias: bool
    
    :return out: None
    :type out: None
    """
    super(LinearLayer, self).__init__()
    
    self.in_features = in_features
    self.out_features = out_features
    self.num_particles = num_particles
    self.bias = bias
    
    self.fc = nn.Linear(self.in_features, self.out_features, bias=self.bias)

    
  def forward(self, h: Tensor) -> Tensor:
    """The call method of the layer 
    :param h: Input tensor containing the propagated features from the previous layer
    :type h: class: `torch.Tensor`
        
    :return out: Output tensor containing the output features from the current layer
                 after being pass through `torch.nn.Linear` and the layer
                 corresponding non-linear activation function.
    :rtype out: class: `torch.Tensor`
    """
    return self.fc(h)