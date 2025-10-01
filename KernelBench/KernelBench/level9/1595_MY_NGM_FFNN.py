import random
import torch
import torch.nn as nn
from collections import defaultdict
import torch.nn.functional as F
import torch.optim as optim


class MY_NGM_FFNN(nn.Module):

    def __init__(self, alpha, input_dim, hidden1_dim, hidden2_dim,
        output_dim, device=torch.device('cpu')):
        super(MY_NGM_FFNN, self).__init__()
        self.hidden1 = nn.Linear(input_dim, hidden1_dim)
        self.hidden2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.output = nn.Linear(hidden2_dim, output_dim)
        self.alpha = alpha
        self.device = device
        self

    def save(self, output_dir, model_name):
        None
        torch.save(self.state_dict(), output_dir + model_name + '.pt')
        None

    def load(self, output_dir, model_name):
        None
        self.load_state_dict(torch.load(output_dir + model_name + '.pt'))
        None

    def forward(self, tf_idf_vec):
        hidden1 = F.relu(self.hidden1(tf_idf_vec))
        hidden2 = F.relu(self.hidden2(hidden1))
        return F.log_softmax(self.output(hidden2), -1)

    def reset_parameters(self):
        self.hidden1.reset_parameters()
        self.hidden2.reset_parameters()
        self.output.reset_parameters()

    def get_last_hidden(self, tf_idf_vec):
        hidden1 = F.relu(self.hidden1(tf_idf_vec))
        return F.relu(self.hidden2(hidden1))

    def train_(self, seed_nodes, train_node_pairs, node2vec, node2label,
        num_epoch, batch_size, learning_rate):
        None
        self.train()
        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        node2neighbors = defaultdict(list)
        for src, dest in train_node_pairs:
            node2neighbors[src].append(dest)
            node2neighbors[dest].append(src)
        labeled_nodes = dict()
        for node in seed_nodes:
            labeled_nodes[node] = node2label[node]
        iteration = 1
        while iteration < 2:
            None
            None
            iteration += 1
            for e in range(NUM_EPOCH):
                train_node_pairs_cpy = train_node_pairs[:]
                total_loss = 0
                count = 0
                while train_node_pairs_cpy:
                    optimizer.zero_grad()
                    loss = torch.tensor(0, dtype=torch.float32, device=self
                        .device)
                    try:
                        batch = random.sample(train_node_pairs_cpy, batch_size)
                    except ValueError:
                        break
                    for src, dest in batch:
                        count += 1
                        train_node_pairs_cpy.remove((src, dest))
                        src_vec = torch.tensor(node2vec[src])
                        dest_vec = torch.tensor(node2vec[dest])
                        if src in labeled_nodes:
                            src_target = torch.tensor([labeled_nodes[src]])
                            src_softmax = self.forward(torch.tensor(src_vec))
                            src_incident_edges = len(node2neighbors[src])
                            loss += loss_function(src_softmax.view(1, -1),
                                src_target) * (1 / src_incident_edges)
                        if dest in labeled_nodes:
                            dest_target = torch.tensor([labeled_nodes[dest]])
                            dest_softmax = self.forward(torch.tensor(dest_vec))
                            dest_incident_edges = len(node2neighbors[dest])
                            loss += loss_function(dest_softmax.view(1, -1),
                                dest_target) * (1 / dest_incident_edges)
                        loss += self.alpha * torch.dist(self.
                            get_last_hidden(src_vec), self.get_last_hidden(
                            dest_vec))
                    if loss.item() != 0:
                        assert not torch.isnan(loss)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                        del loss
                total_loss / len(labeled_nodes)
                None
            for node in list(labeled_nodes.keys()):
                label = labeled_nodes[node]
                for neighbor in node2neighbors[node]:
                    if neighbor not in labeled_nodes:
                        labeled_nodes[neighbor] = label

    def predict(self, tf_idf_vec):
        return torch.argmax(self.forward(tf_idf_vec)).item()

    def evaluate(self, test_nodes, node2vec, node2label):
        self.eval()
        None
        correct_count = 0
        for node in test_nodes:
            predicted = self.predict(torch.tensor(node2vec[node], device=
                self.device))
            None
            if predicted == node2label[node]:
                correct_count += 1
        return float(correct_count) / len(test_nodes)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'alpha': 4, 'input_dim': 4, 'hidden1_dim': 4,
        'hidden2_dim': 4, 'output_dim': 4}]
