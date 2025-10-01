import torch


class Attention(torch.nn.Module):
    """
    attention_size_1: Number of neurons in 1st attention layer.
    attention_size_2: Number of neurons in 2nd attention layer.        
    """

    def __init__(self, attention_size_1, attention_size_2):
        super(Attention, self).__init__()
        self.attention_1 = torch.nn.Linear(attention_size_1, attention_size_2)
        self.attention_2 = torch.nn.Linear(attention_size_2, attention_size_1)
    """
    Forward propagation pass
    
    gets x_in: Primary capsule output
    condensed_x: Attention normalized capsule output
    
    """

    def forward(self, x_in):
        attention_score_base = self.attention_1(x_in)
        attention_score_base = torch.nn.functional.relu(attention_score_base)
        attention_score = self.attention_2(attention_score_base)
        attention_score = torch.nn.functional.softmax(attention_score, dim=0)
        condensed_x = x_in * attention_score
        return condensed_x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'attention_size_1': 4, 'attention_size_2': 4}]
