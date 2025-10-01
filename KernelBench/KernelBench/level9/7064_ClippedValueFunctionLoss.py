from torch.nn import Module
import torch
import torch.utils.data
import torch.nn.functional
import torch.autograd


class ClippedValueFunctionLoss(Module):
    """
    ## Clipped Value Function Loss

    Similarly we clip the value function update also.

    egin{align}
    V^{\\pi_	heta}_{CLIP}(s_t)
     &= clip\\Bigl(V^{\\pi_	heta}(s_t) - \\hat{V_t}, -\\epsilon, +\\epsilon\\Bigr)
    \\
    \\mathcal{L}^{VF}(	heta)
     &= rac{1}{2} \\mathbb{E} iggl[
      max\\Bigl(igl(V^{\\pi_	heta}(s_t) - R_tigr)^2,
          igl(V^{\\pi_	heta}_{CLIP}(s_t) - R_tigr)^2\\Bigr)
     iggr]
    \\end{align}

    Clipping makes sure the value function $V_	heta$ doesn't deviate
     significantly from $V_{	heta_{OLD}}$.

    """

    def forward(self, value: 'torch.Tensor', sampled_value: 'torch.Tensor',
        sampled_return: 'torch.Tensor', clip: 'float'):
        clipped_value = sampled_value + (value - sampled_value).clamp(min=-
            clip, max=clip)
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value -
            sampled_return) ** 2)
        return 0.5 * vf_loss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
