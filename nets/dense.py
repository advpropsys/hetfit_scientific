from torch import nn

class Net(nn.Module):
    def __init__(self,input_dim:int=2,hidden_dim:int=200):
        super(Net,self).__init__()
        self.input = nn.Linear(input_dim,40)
        self.act1 = nn.Tanh()
        self.layer = nn.Linear(40,80)
        self.act2 = nn.ReLU()
        self.layer1 = nn.Linear(80,hidden_dim)
        self.act3 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim,1)
        
    def forward(self, x):
        x = self.act2(self.layer(self.act1(self.input(x))))
        x = self.act3(self.layer1(x))
        x = self.layer2(x)
        return x
