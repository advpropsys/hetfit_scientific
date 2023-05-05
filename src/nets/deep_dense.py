from torch import nn
from torch.functional import F

class dmodel(nn.Module):
    def __init__(self, in_features=1, hidden_features=200, out_features=1):
        """
        We're creating a neural network with 4 layers, each with 200 neurons. The first layer takes in
        the input, the second layer takes in the output of the first layer, the third layer takes in the
        output of the second layer, and the fourth layer takes in the output of the third layer
        
        :param in_features: The number of input features, defaults to 1 (optional)
        :param hidden_features: the number of neurons in the hidden layers, defaults to 200 (optional)
        :param out_features: The number of classes for classification (1 for regression), defaults to 1
        (optional)
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