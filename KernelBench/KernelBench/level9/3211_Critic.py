import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):

    def __init__(self, dim_input, dim_output):
        super(Critic, self).__init__()
        self._dim_input = dim_input
        self._dim_output = dim_output
        H_LAYER1 = 50
        H_LAYER2 = 20
        self.linear1 = nn.Linear(self._dim_input, H_LAYER1)
        self.linear2 = nn.Linear(H_LAYER1, H_LAYER2)
        self.linear3 = nn.Linear(H_LAYER2, self._dim_output)

    def forward(self, s, a):
        """
        s = Variable(torch.FloatTensor(np.array(s,dtype=np.float32)))
        if(type(a)!=type(s)):
            a = Variable(torch.FloatTensor(np.array(a,dtype=np.float32)))
        """
        x = torch.cat([s, a], 1)
        a1 = F.relu(self.linear1(x))
        a2 = F.relu(self.linear2(a1))
        y = self.linear3(a2)
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_input': 4, 'dim_output': 4}]
