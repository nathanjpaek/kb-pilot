import torch


def masked_softmax(x, m=None, axis=-1):
    """
    Softmax with mask (optional)
    """
    x = torch.clamp(x, min=-15.0, max=15.0)
    if m is not None:
        m = m.float()
        x = x * m
    e_x = torch.exp(x - torch.max(x, dim=axis, keepdim=True)[0])
    if m is not None:
        e_x = e_x * m
    softmax = e_x / (torch.sum(e_x, dim=axis, keepdim=True) + 1e-06)
    return softmax


class BoundaryDecoderAttention(torch.nn.Module):
    """
        input:  p:          batch x inp_p
                p_mask:     batch
                q:          batch x time x inp_q
                q_mask:     batch x time
                h_tm1:      batch x out
                depth:      int
        output: z:          batch x inp_p+inp_q
    """

    def __init__(self, input_dim, output_dim, enable_cuda=False):
        super(BoundaryDecoderAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.enable_cuda = enable_cuda
        self.V = torch.nn.Linear(self.input_dim, self.output_dim)
        self.W_a = torch.nn.Linear(self.output_dim, self.output_dim)
        self.v = torch.nn.Parameter(torch.FloatTensor(self.output_dim))
        self.c = torch.nn.Parameter(torch.FloatTensor(1))
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform(self.V.weight.data, gain=1)
        torch.nn.init.xavier_uniform(self.W_a.weight.data, gain=1)
        self.V.bias.data.fill_(0)
        self.W_a.bias.data.fill_(0)
        torch.nn.init.normal(self.v.data, mean=0, std=0.05)
        self.c.data.fill_(1.0)

    def forward(self, H_r, mask_r, h_tm1):
        batch_size, time = H_r.size(0), H_r.size(1)
        Fk = self.V.forward(H_r.view(-1, H_r.size(2)))
        Fk_prime = self.W_a.forward(h_tm1)
        Fk = Fk.view(batch_size, time, -1)
        Fk = torch.tanh(Fk + Fk_prime.unsqueeze(1))
        beta = torch.matmul(Fk, self.v)
        beta = beta + self.c.unsqueeze(0)
        beta = masked_softmax(beta, mask_r, axis=-1)
        z = torch.bmm(beta.view(beta.size(0), 1, beta.size(1)), H_r)
        z = z.view(z.size(0), -1)
        return z, beta


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4}]
