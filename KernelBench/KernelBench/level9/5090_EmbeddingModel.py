import torch
import torch.nn.functional as F
from torch import nn
from torch import optim


class EmbeddingModel(nn.Module):

    def __init__(self, obs_size, num_outputs):
        super(EmbeddingModel, self).__init__()
        self.obs_size = obs_size
        self.num_outputs = num_outputs
        self.fc1 = nn.Linear(obs_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.last = nn.Linear(32 * 2, num_outputs)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-05)

    def forward(self, x1, x2):
        x1 = self.embedding(x1)
        x2 = self.embedding(x2)
        x = torch.cat([x1, x2], dim=2)
        x = self.last(x)
        return nn.Softmax(dim=2)(x)

    def embedding(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def train_model(self, batch):
        batch_size = torch.stack(batch.state).size()[0]
        states = torch.stack(batch.state).view(batch_size, config.
            sequence_length, self.obs_size)[:, -5:, :]
        next_states = torch.stack(batch.next_state).view(batch_size, config
            .sequence_length, self.obs_size)[:, -5:, :]
        actions = torch.stack(batch.action).view(batch_size, config.
            sequence_length, -1).long()[:, -5:, :]
        self.optimizer.zero_grad()
        net_out = self.forward(states, next_states)
        actions_one_hot = torch.squeeze(F.one_hot(actions, self.num_outputs)
            ).float()
        loss = nn.MSELoss()(net_out, actions_one_hot)
        loss.backward()
        self.optimizer.step()
        return loss.item()


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'obs_size': 4, 'num_outputs': 4}]
