from torch import nn,tensor
    
class PINNd_p(nn.Module):
    def __init__(self):
        super(PINNd_p,self).__init__()
        weights = tensor([60.])
        self.weights = nn.Parameter(weights)
    def forward(self,x):
        c, = self.weights
        x1 = c*x**0.5
        return x1
    
class PINNhd_ma(nn.Module):
    def __init__(self):
        super(PINNd_p,self).__init__()
        weights = tensor([0.01])
        self.weights = nn.Parameter(weights)
    def forward(self,x):
        c, = self.weights
        x1 = c*x[0]*x[1]
        return x1
    
    