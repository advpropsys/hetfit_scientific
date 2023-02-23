from torch import nn,tensor
import numpy as np
import seaborn as sns
class PINNd_p(nn.Module):
    """ $d \mapsto P$

    
    """
    def __init__(self):
        super(PINNd_p,self).__init__()
        weights = tensor([60.,0.5])
        self.weights = nn.Parameter(weights)
    def forward(self,x):
        """P,U input, d output

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        c,b = self.weights
        x1 = (x[0]/(c*x[1]))**0.5
        return x1
    
class PINNhd_ma(nn.Module):
    """ $h,d \mapsto m_a $

    
    """
    def __init__(self):
        super(PINNhd_ma,self).__init__()
        weights = tensor([0.01])
        self.weights = nn.Parameter(weights)
    def forward(self,x):
        c, = self.weights
        x1 = c*x[0]*x[1]
        return x1
    
class PINNT_ma(nn.Module):
    """$ m_a, U \mapsto T$

   
    """
    def __init__(self):
        super(PINNT_ma,self).__init__()
        weights = tensor([0.01])
        self.weights = nn.Parameter(weights)
    def forward(self,x):
        c, = self.weights
        x1 = c*x[0]*x[1]**0.5
        return x1
    
    
    
    
    

    
    