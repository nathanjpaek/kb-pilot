import torch


def squash(s, axis=-1, epsilon=1e-07):
    squared_norm = torch.sum(s * s, dim=axis)
    safe_norm = torch.sqrt(squared_norm + epsilon)
    squash_factor = squared_norm / (1.0 + squared_norm)
    unit_vector = torch.div(s, safe_norm.unsqueeze(-1))
    return torch.mul(squash_factor.unsqueeze(-1), unit_vector)


def safe_norm(s, axis=-1, epsilon=1e-07):
    squared_norm = torch.mul(s, s).sum(dim=axis)
    return torch.sqrt(squared_norm + epsilon)


class Agreement_Routing_Down(torch.nn.Module):
    """This is the localised agreement routing algorithm. It takes in the total
    prediction vectors from a layer l and computes the routing weights for 
    those predictions. It then squashes the prediction vectors using the 
    custom squash function."""

    def __init__(self, bias, input_caps_maps, input_caps_dim,
        output_caps_maps, output_caps_dim, new_hl, new_wl, num_iterations):
        super(Agreement_Routing_Down, self).__init__()
        self.input_caps_maps = input_caps_maps
        self.input_caps_dim = input_caps_dim
        self.output_caps_maps = output_caps_maps
        self.output_caps_dim = output_caps_dim
        self.new_hl = int(new_hl)
        self.new_wl = int(new_wl)
        self.num_iterations = num_iterations
        self.softmax = torch.nn.Softmax(dim=-1)
        self.b = torch.nn.Parameter(torch.zeros((1, self.output_caps_maps,
            self.new_hl, self.new_wl, self.input_caps_maps)))

    def forward(self, tensor_of_prediction_vector):
        c = self.softmax(self.b)
        output_vectors = torch.mul(c.unsqueeze(-1), tensor_of_prediction_vector
            )
        output_vectors = output_vectors.sum(dim=-2)
        output_vectors = squash(output_vectors, axis=-1)
        b_batch = self.b
        for d in range(self.num_iterations):
            b_batch = b_batch + torch.mul(tensor_of_prediction_vector,
                output_vectors.unsqueeze(-2)).sum(dim=-1)
            """
            distances = torch.mul(tensor_of_prediction_vector,
                                output_vectors.unsqueeze(-2)).sum(dim = -1)
            
            self.b = torch.add(self.b, distances)
            """
            c = self.softmax(b_batch)
            output_vectors = torch.mul(tensor_of_prediction_vector, c.
                unsqueeze(-1))
            output_vectors = output_vectors.sum(-2)
            output_vectors = squash(output_vectors, axis=-1)
        self.c = c
        return output_vectors


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'bias': 4, 'input_caps_maps': 4, 'input_caps_dim': 4,
        'output_caps_maps': 4, 'output_caps_dim': 4, 'new_hl': 4, 'new_wl':
        4, 'num_iterations': 4}]
