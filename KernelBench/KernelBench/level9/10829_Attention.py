import torch


def activation_func(name):
    name = name.lower()
    if name == 'sigmoid':
        return torch.nn.Sigmoid()
    elif name == 'tanh':
        return torch.nn.Tanh()
    elif name == 'relu':
        return torch.nn.ReLU()
    elif name == 'softmax':
        return torch.nn.Softmax()
    elif name == 'leaky_relu':
        return torch.nn.LeakyReLU(0.1)
    else:
        return torch.nn.Sequential()


def cosine_similarity(input1, input2):
    query_norm = torch.sqrt(torch.sum(input1 ** 2 + 1e-05, 1))
    doc_norm = torch.sqrt(torch.sum(input2 ** 2 + 1e-05, 1))
    prod = torch.sum(torch.mul(input1, input2), 1)
    norm_prod = torch.mul(query_norm, doc_norm)
    cos_sim_raw = torch.div(prod, norm_prod)
    return cos_sim_raw


class Attention(torch.nn.Module):

    def __init__(self, n_k, activation='relu'):
        super(Attention, self).__init__()
        self.n_k = n_k
        self.fc_layer = torch.nn.Linear(self.n_k, self.n_k, activation_func
            (activation))
        self.soft_max_layer = torch.nn.Softmax()

    def forward(self, pu, mp):
        expanded_pu = pu.repeat(1, len(mp)).view(len(mp), -1)
        inputs = cosine_similarity(expanded_pu, mp)
        fc_layers = self.fc_layer(inputs)
        attention_values = self.soft_max_layer(fc_layers)
        return attention_values


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 16])]


def get_init_inputs():
    return [[], {'n_k': 4}]
