import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as I


class Network(nn.Module):
    """
    Q-network
    """

    def __init__(self, state_size, action_size, seed, fc1_units=64,
        fc2_units=32):
        """
        Build model and Intialize it
        
        Params
        ======
        state_size (int) : State space size
        action_size (int) : Action space size
        seed (int) : Random seed
        fc1_unit (int)  
        fc2_unit (int)
        """
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters of the layers
        xavier_normal is used.
        See "Understanding the difficulty of training deep feedforward neural networks" - Glorot, X. & Bengio, Y. (2010) for details.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                I.xavier_normal_(m.weight)

    def forward(self, state):
        """
        Forward pass state -> action

        Params
        ======
        state (Torch Tensor) [batch_size, state_size]: state vector
            
        Returns
        ======
        actions (Torch Tensor) [batch_size, action_size]: action values
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = F.relu(self.fc3(x))
        return actions


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_size': 4, 'action_size': 4, 'seed': 4}]
