import torch.nn as nn
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal, Normal
from torch.distributions.distribution import Distribution
import numpy as np

class GaussianKDE(Distribution):
    def __init__(self, X, bw):
        """
        X : tensor (n, d)
          `n` points with `d` dimensions to which KDE will be fit
        bw : numeric
          bandwidth for Gaussian kernel
        """
        self.X = X
        self.bw = bw
        self.dims = X.shape[-1]
        self.n = X.shape[0]
        self.mvn = MultivariateNormal(loc=torch.zeros(self.dims),
                                      covariance_matrix=torch.eye(self.dims))

    def sample(self, num_samples):
        idxs = (np.random.uniform(0, 1, num_samples) * self.n).astype(int)
        norm = Normal(loc=self.X[idxs], scale=self.bw)
        return norm.sample()

    def score_samples(self, Y, X=None):
        """Returns the kernel density estimates of each point in `Y`.

        Parameters
        ----------
        Y : tensor (m, d)
          `m` points with `d` dimensions for which the probability density will
          be calculated
        X : tensor (n, d), optional
          `n` points with `d` dimensions to which KDE will be fit. Provided to
          allow batch calculations in `log_prob`. By default, `X` is None and
          all points used to initialize KernelDensityEstimator are included.


        Returns
        -------
        log_probs : tensor (m)
          log probability densities for each of the queried points in `Y`
        """
        if X == None:
            X = self.X
        log_probs = self.mvn.log_prob((X.unsqueeze(1) - Y)).sum()

        return log_probs

    def log_prob(self, Y):
        """Returns the total log probability of one or more points, `Y`, using
        a Multivariate Normal kernel fit to `X` and scaled using `bw`.

        Parameters
        ----------
        Y : tensor (m, d)
          `m` points with `d` dimensions for which the probability density will
          be calculated

        Returns
        -------
        log_prob : numeric
          total log probability density for the queried points, `Y`
        """

        X_chunks = self.X
        Y_chunks = Y
        self.Y = Y
        log_prob = 0

        for x in X_chunks:
            for y in Y_chunks:
                
                log_prob += self.score_samples(y,x).sum(dim=0)

        return log_prob
    
class PlanarFlow(nn.Module):
    """
    A single planar flow, computes T(x) and log(det(jac_T)))
    """
    def __init__(self, D):
        super(PlanarFlow, self).__init__()
        self.u = nn.Parameter(torch.Tensor(1, D), requires_grad=True)
        self.w = nn.Parameter(torch.Tensor(1, D), requires_grad=True)
        self.b = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.h = torch.tanh
        self.init_params()

    def init_params(self):
        self.w.data.uniform_(-1, 1)
        self.b.data.uniform_(-1, 1)
        self.u.data.uniform_(-1, 1)

    def forward(self, z):
        linear_term = torch.mm(z, self.w.T) + self.b
        return z + self.u * self.h(linear_term)

    def h_prime(self, x):
        """
        Derivative of tanh
        """
        return (1 - self.h(x) ** 2)

    def psi(self, z):
        inner = torch.mm(z, self.w.T) + self.b
        return self.h_prime(inner) * self.w

    def log_det(self, z):
        inner = 1 + torch.mm(self.psi(z), self.u.T)
        return torch.log(torch.abs(inner))


class NormalizingFlow(nn.Module):
    """
    A normalizng flow composed of a sequence of planar flows.
    """
    def __init__(self, D, n_flows=2):
        super(NormalizingFlow, self).__init__()
        self.flows = nn.ModuleList(
            [PlanarFlow(D) for _ in range(n_flows)])

    def sample(self, base_samples):
        """
        Transform samples from a simple base distribution
        by passing them through a sequence of Planar flows.
        """
        samples = base_samples
        for flow in self.flows:
            samples = flow(samples)
        return samples

    def forward(self, x):
        """
        Computes and returns the sum of log_det_jacobians
        and the transformed samples T(x).
        """
        sum_log_det = 0
        transformed_sample = x

        for i in range(len(self.flows)):
            log_det_i = (self.flows[i].log_det(transformed_sample))
            sum_log_det += log_det_i
            transformed_sample = self.flows[i](transformed_sample)

        return transformed_sample, sum_log_det
    
def random_normal_samples(n, dim=2):
    return torch.zeros(n, dim).normal_(mean=0, std=1)



class nflow():
    def __init__(self,dim=2,latent=16,batchsize:int=1):
        self.dim = dim
        self.batchsize = batchsize
        self.model = NormalizingFlow(dim, latent)

    def compile(self,optim:torch.optim=torch.optim.Adam,lr:float=0.0001):
        self.opt = optim(
            params=self.model.parameters(),
            lr=lr,
            # momentum=0.1
        )
        scaler = StandardScaler()
        
        df = pd.read_csv('data/dataset.csv')
        df = df.iloc[:,1:]
        self.scaled = scaler.fit_transform(df)
        self.density = GaussianKDE(X=torch.tensor(self.scaled, dtype=torch.float32), bw=0.03)
        
        # self.dl = torch.utils.data.DataLoader(scaled,batchsize=self.batchsize)
        self.scheduler = ReduceLROnPlateau(self.opt, 'min', patience=1000)
        self.losses = []

    def train(self,iters:int=1000):
        for idx in range(iters):
            if idx % 100 == 0:
                print("Iteration {}".format(idx))

            samples = torch.autograd.Variable(random_normal_samples(self.batchsize,self.dim))

            z_k, sum_log_det = self.model(samples)
            log_p_x = self.density.log_prob(z_k)
            # Reverse KL since we can evaluate target density but can't sample
            loss = (- sum_log_det - (log_p_x)).mean()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.scheduler.step(loss)

            self.losses.append(loss.item())

            if idx % 100 == 0:
                print("Loss {}".format(loss.item()))
                
        plt.plot(self.losses)