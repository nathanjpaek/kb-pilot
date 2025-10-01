import torch
from torch import nn


class PointerNetwork(nn.Module):

    def __init__(self, input_size, model_dim, attn_size=75, dropout=0.2):
        """ Pointer Network
                Args:
                    input_size(int): size of input
                Input:
                    - **H** of shape `(passage_legth, batch, input_size)`: a float tensor in which we determine
                    the importance of information in the passage regarding a question
                    - **question** of shape `(question_length, batch, input_size)`: a float tensor containing question
                    representation
                Output:
                    - start(torch.tensor of shape (batch_size, passage_length, 1)): start position of the answer
                    - end(torch.tensor of shape (batch_size, passage_length, 1)): end position of the answer

        """
        super(PointerNetwork, self).__init__()
        self.Whp = nn.Linear(input_size, attn_size, bias=False)
        self.Wha1 = nn.Linear(model_dim, attn_size, bias=False)
        self.v = nn.Linear(attn_size, 1, bias=False)
        self.cell = nn.GRUCell(input_size, model_dim, False)
        self.Wuq = nn.Linear(model_dim, attn_size, bias=False)
        self.v1 = nn.Linear(attn_size, 1)

    def get_initial_state(self, question):
        s = torch.tanh(self.Wuq(question))
        s = self.v1(s)
        a = nn.functional.softmax(s, 0)
        r = a * question
        return r.sum(0)

    def forward(self, h, question):
        h_a = self.get_initial_state(question)
        Wh = self.Whp(h)
        s = torch.tanh(Wh + self.Wha1(h_a))
        s = self.v(s)
        p1 = nn.functional.softmax(s, 0)
        start = nn.functional.log_softmax(s, 0).transpose(0, 1)
        c = (p1 * h).sum(0)
        h_a = self.cell(c, h_a)
        s = torch.tanh(Wh + self.Wha1(h_a))
        s = self.v(s)
        end = nn.functional.log_softmax(s, 0).transpose(0, 1)
        return start.squeeze(), end.squeeze()


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'model_dim': 4}]
