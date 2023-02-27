from torch import nn


class Net(nn.Module):
    """The Net class inherits from the nn.Module class, which has a number of attributes and methods (such
    as .parameters() and .zero_grad()) which we will be using. You can read more about the nn.Module
    class [here](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)"""
    def __init__(self,input_dim:int=2,hidden_dim:int=200):
        """
        We create a neural network with two hidden layers, each with **hidden_dim** neurons, and a ReLU activation
        function. The output layer has one neuron and no activation function
        
        :param input_dim: The dimension of the input, defaults to 2
        :type input_dim: int (optional)
        :param hidden_dim: The number of neurons in the hidden layer, defaults to 200
        :type hidden_dim: int (optional)
        """
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
