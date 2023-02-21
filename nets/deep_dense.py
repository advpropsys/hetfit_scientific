from torch import nn
from torch.functional import F

class dmodel(nn.Module):
    def __init__(self, in_features=1, hidden_features=200, out_features=1):
        super(dmodel, self).__init__()
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, hidden_features)
        self.fc4 = nn.Linear(hidden_features, out_features)
                
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x) # ReLU activation
        x = self.fc2(x)
        x = F.relu(x) # ReLU activation
        x = self.fc3(x)
        x = F.relu(x) # ReLU activation
        x = self.fc4(x)
        return x