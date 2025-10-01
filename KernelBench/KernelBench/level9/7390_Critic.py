import torch


class Critic(torch.nn.Module):

    def __init__(self, critic_lr, critic_epochs):
        super(Critic, self).__init__()
        self.initialize_network()
        self.optimizer = torch.optim.Adam(lr=critic_lr, params=self.
            parameters())
        self.loss = torch.nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else
            'cpu:0')
        self

    def initialize_network(self):
        self.fc1 = torch.nn.Linear(1024, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 1)
        self.relu = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=4, stride=2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=4, stride=2)

    def forward(self, x):
        out = torch.Tensor(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.relu(out)
        out = out.reshape(-1, 1024)
        out = self.fc1(out)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        out = self.relu(out)
        return out

    def optimize(self, states, rewards, epochs, batch_sz):
        n_samples = rewards.shape[0]
        num_batch = int(n_samples // batch_sz)
        for i in tqdm(range(epochs)):
            for b in range(num_batch):
                s = states[b * batch_sz:(b + 1) * batch_sz]
                r = rewards[b * batch_sz:(b + 1) * batch_sz]
                p = self.forward(s)
                loss = self.loss(p, r)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            s = states[num_batch * batch_sz:]
            r = rewards[num_batch * batch_sz:]
            p = self.forward(s)
            loss = self.loss(p, r)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {'critic_lr': 4, 'critic_epochs': 4}]
