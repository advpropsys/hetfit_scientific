from torch import nn
from torch.functional import F

class dmodel(nn.Module):
    """4 layers Torch model. Relu activations, hidden layers are same size.

    """
    def __init__(self, in_features=1, hidden_features=200, out_features=1):
        """Init

        Args:
            in_features (int, optional): Input features. Defaults to 1.
            hidden_features (int, optional): Hidden dims. Defaults to 200.
            out_features (int, optional): Output dims. Defaults to 1.
        """
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