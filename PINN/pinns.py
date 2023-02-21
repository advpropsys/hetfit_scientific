from torch import nn,tensor
    
class PINNlow(nn.Module):
    def __init__(self):
        super(PINNlow,self).__init__()
        weights = tensor([60.])
        self.weights = nn.Parameter(weights)
    def forward(self,x):
        c, = self.weights
        x1 = c*x**0.5
        return x1
    
    