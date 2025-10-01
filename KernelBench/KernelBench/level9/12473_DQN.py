import torch
import torch.nn.functional as F
import torch.nn as nn


class DQN(nn.Module):
    """Initialize a deep Q-learning network

    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    This is just a hint. You can build your own structure.
    """

    def __init__(self, gamma, ddqn=False, in_channels=4, num_actions=4):
        """
        Parameters:
        -----------
        in_channels: number of channel of input.
                i.e The number of most recent frames stacked together, here we use 4 frames, which means each state in Breakout is composed of 4 frames.
        num_actions: number of action-value to output, one-to-one correspondence to action in game.

        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(DQN, self).__init__()
        self.gamma = gamma
        self.ddqn = ddqn
        self.in_channels = in_channels
        self.num_actions = num_actions
        self.pool = nn.MaxPool2d(2, 2)
        self.conv_1 = nn.Conv2d(self.in_channels, 32, kernel_size=8, stride=4)
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc_1 = nn.Linear(64 * 7 * 7, 512)
        self.output_layer = nn.Linear(512, self.num_actions)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        """Execute a forward propagation step for the neural network.

        Args:
            x: An observation in the form of a (4, 84, 84) tensor.
        """
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
        x = x.reshape(-1, 64 * 7 * 7)
        x = F.relu(self.fc_1(x))
        x = self.output_layer(x)
        return x

    def compute_loss(self, tensor_lst, target_network, criterion):
        """Computes loss between target-Q function and current Q-function.

        Args:
            tensor_lst: A list of 5 tensors - current states, current actions, current rewards,
                terminal state booleans, and next states.

        Returns:
            Loss values in the form of a PyTorch tensor.
        """
        obs, act, rew, done, next_obs = tensor_lst
        with torch.no_grad():
            if self.ddqn:
                next_state_training_q_vals = self(next_obs)
                max_training_q_vals_ind = next_state_training_q_vals.argmax(dim
                    =1, keepdim=True)
                next_state_target_q_vals = target_network(next_obs)
                max_target_q_vals = torch.gather(input=
                    next_state_target_q_vals, dim=1, index=
                    max_training_q_vals_ind)
                target_q_vals = rew + self.gamma * (1 - done
                    ) * max_target_q_vals
            else:
                next_state_target_q_vals = target_network(next_obs)
                max_target_q_vals = next_state_target_q_vals.max(dim=1,
                    keepdim=True)[0]
                target_q_vals = rew + self.gamma * (1 - done
                    ) * max_target_q_vals
        q_vals = self(obs)
        actual_q_vals = torch.gather(input=q_vals, dim=1, index=act)
        return criterion(actual_q_vals, target_q_vals)


def get_inputs():
    return [torch.rand([4, 4, 144, 144])]


def get_init_inputs():
    return [[], {'gamma': 4}]
