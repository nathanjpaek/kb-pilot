import torch
import torch.nn.functional as F
from torch import nn


class Encoder(nn.Module):

    def __init__(self, input_size: 'int', output_size: 'int', max_temp:
        'float'=10.0, min_temp: 'float'=0.1, reg_threshold: 'float'=3.0,
        reg_eps: 'float'=1e-10) ->None:
        """Feature selection encoder
        Implemented according to "`Concrete Autoencoders for Differentiable Feature Selection and Reconstruction.`"
        :cite:p:`DBLP:journals/corr/abs-1901-09346`.

        Args:
            input_size: size of the input layer. Should be the same as the `output_size` of the decoder.
            output_size: size of the latent layer. Should be the same as the `input_size` of the decoder.
            max_temp: maximum temperature for Gumble Softmax. Defaults to 10.0.
            min_temp: minimum temperature for Gumble Softmax. Defaults to 0.1.
            reg_threshold: regularization threshold. The encoder will be penalized when the sum of
                probabilities for a selection neuron exceed this threshold. Defaults to 0.3.
            reg_eps: regularization epsilon. Minimum value for the clamped softmax function in
                regularization term. Defaults to 1e-10.
        """
        super(Encoder, self).__init__()
        self.register_buffer('temp', torch.tensor(max_temp))
        self.register_buffer('max_temp', torch.tensor(max_temp))
        self.register_buffer('min_temp', torch.tensor(min_temp))
        self.register_buffer('reg_threshold', torch.tensor(reg_threshold))
        self.register_buffer('reg_eps', torch.tensor(reg_eps))
        logits = nn.init.xavier_normal_(torch.empty(output_size, input_size))
        self.logits = nn.Parameter(logits, requires_grad=True)

    @property
    def latent_features(self):
        return torch.argmax(self.logits, 1)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Uses the trained encoder to make inferences.

        Args:
            x (torch.Tensor): input data. Should be the same size as the encoder input.

        Returns:
            torch.Tensor: encoder output of size `output_size`.
        """
        logits_size = self.logits.size()
        if self.training:
            uniform = torch.rand(logits_size, device=x.device)
            gumbel = -torch.log(-torch.log(uniform))
            noisy_logits = (self.logits + gumbel) / self.temp
            samples = F.softmax(noisy_logits, dim=1)
            selections = samples
        else:
            dim_argmax = len(logits_size) - 1
            logits_argmax = torch.argmax(self.logits, dim_argmax)
            discrete_logits = F.one_hot(logits_argmax, num_classes=
                logits_size[1])
            selections = discrete_logits
        encoded = torch.matmul(x, torch.transpose(selections.float(), 0, 1))
        return encoded

    def update_temp(self, current_epoch, max_epochs) ->torch.Tensor:
        self.temp = self.max_temp * torch.pow(self.min_temp / self.max_temp,
            current_epoch / max_epochs)
        return self.temp

    def calc_mean_max(self) ->torch.Tensor:
        logits_softmax = F.softmax(self.logits, dim=1)
        logits_max = torch.max(logits_softmax, 1).values
        mean_max = torch.mean(logits_max)
        return mean_max

    def regularization(self) ->float:
        """Regularization term according to https://homes.esat.kuleuven.be/~abertran/reports/TS_JNE_2021.pdf. The sum of
        probabilities for a selection neuron is penalized if its larger than the threshold value. The returned value is
        summed with the loss function."""
        selection = torch.clamp(F.softmax(self.logits, dim=1), self.reg_eps, 1)
        return torch.sum(F.relu(torch.norm(selection, 1, dim=0) - self.
            reg_threshold))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4}]
