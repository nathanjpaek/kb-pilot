import torch
import torch.nn.functional as F
import torch.nn as nn


class SharedAgent(torch.nn.Module):
    """
    A simple two headed / chimera Actor Critic agent.
    The actor and critic share the body of the network.
    It is argued that this is because "good" actions 
    correlate to visiting states with "large" values, and
    so there should exist some form of shared information 
    between these two functions, thus motivating the shared 
    body.  However, I haven't seen a rigorous proof of this, 
    and training an AC model with a shared body usually just 
    leads to added complications in my experience.  If you
    know a good reference for a mathematical proof on why 
    this should be done please let me know!
    """

    def __init__(self, numObs, numActions, numHidden):
        super(SharedAgent, self).__init__()
        self.shared_input = nn.Linear(numObs, numHidden)
        self.shared_fc1 = nn.Linear(numHidden, numHidden)
        self.shared_fc2 = nn.Linear(numHidden, 2 * numHidden)
        self.actor_output = nn.Linear(2 * numHidden, numActions)
        self.critic_output = nn.Linear(2 * numHidden, 1)

    def forward(self, x):
        x = F.relu(self.shared_input(x))
        x = F.relu(self.shared_fc1(x))
        x = F.relu(self.shared_fc2(x))
        logits = self.actor_output(x)
        value = self.critic_output(x)
        return logits, value


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'numObs': 4, 'numActions': 4, 'numHidden': 4}]
