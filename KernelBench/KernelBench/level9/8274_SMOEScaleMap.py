import torch
import torch.nn as nn
import torch.autograd


class SMOEScaleMap(nn.Module):
    """
        Compute SMOE Scale on a 4D tensor. This acts as a standard PyTorch layer. 
        
        SMOE Scale is computed independantly for each batch item at each location x,y
    
        Input should be:
        
        (1) A tensor of size [batch x channels x height x width] 
        (2) A tensor with only positive values. (After a ReLU)
        
        Output is a 3D tensor of size [batch x height x width] 
    """

    def __init__(self, run_relu=False):
        super(SMOEScaleMap, self).__init__()
        """
            SMOE Scale must take in values > 0. Optionally, we can run a ReLU to do that.
        """
        if run_relu:
            self.relu = nn.ReLU(inplace=False)
        else:
            self.relu = None

    def forward(self, x):
        assert torch.is_tensor(x), 'input must be a Torch Tensor'
        assert len(x.size()) > 2, 'input must have at least three dims'
        """
            If we do not have a convenient ReLU to pluck from, we can do it here
        """
        if self.relu is not None:
            x = self.relu(x)
        """
            avoid log(0)
        """
        x = x + 1e-07
        """
            This is one form. We can also use the log only form.
        """
        m = torch.mean(x, dim=1)
        k = torch.log2(m) - torch.mean(torch.log2(x), dim=1)
        th = k * m
        return th


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
