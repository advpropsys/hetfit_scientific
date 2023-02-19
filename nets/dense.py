from torch import nn

class Net(nn.Module):
        def __init__(self,dropout = True, hidden_dim = 8):
            self.dropout = dropout
            
            super(self.Net,self).__init__()
            self.input = nn.Linear(2,4)
            self.act1 = nn.Tanh()
            self.layer1 = nn.Linear(4,hidden_dim)
            self.batchnorm = nn.BatchNorm1d(hidden_dim)
            if dropout:
                self.drop = nn.Dropout(p=0.05)
            self.layer2 = nn.Linear(hidden_dim,2)
            
        def forward(self, x):
            x = self.act1(self.input(x))
            x = self.layer1(x)
            x = self.batchnorm(x)
            if self.dropout:
                x = self.drop(x)
            x = self.layer2(x)
            return x