import torch
import torch.nn as nn
import torch.autograd


class GaussNorm2D(nn.Module):
    """
        This will normalize a saliency map to range from 0 to 1 via normal cumulative distribution function. 
        
        Input and output will be a 3D tensor of size [batch size x height x width]. 
        
        Input can be any real valued number (supported by hardware)
        Output will range from 0 to 1
        
        Notes:
        
        (1) GammaNorm2D will produce slightly better results
            The sum ROAR/KAR will improve from 1.44 to 1.45 for FastCAM using GradCAM.
        (2) This method is a bit less expensive than GammaNorm2D.
    """

    def __init__(self, const_mean=None, const_std=None):
        super(GaussNorm2D, self).__init__()
        assert isinstance(const_mean, float) or const_mean is None
        assert isinstance(const_std, float) or const_std is None
        self.const_mean = const_mean
        self.const_std = const_std

    def forward(self, x):
        """
            Input: 
                x:     A Torch Tensor image with shape [batch size x height x width] e.g. [64,7,7]
            Return:
                x:     x Normalized by computing mean and std over each individual batch item and squashed with a 
                       Normal/Gaussian CDF.  
        """
        assert torch.is_tensor(x), 'Input must be a Torch Tensor'
        assert len(x.size()
            ) == 3, 'Input should be sizes [batch size x height x width]'
        s0 = x.size()[0]
        s1 = x.size()[1]
        s2 = x.size()[2]
        x = x.reshape(s0, s1 * s2)
        """
            Compute Mean
        """
        if self.const_mean is None:
            m = x.mean(dim=1)
            m = m.reshape(m.size()[0], 1)
        else:
            m = self.const_mean
        """
            Compute Standard Deviation
        """
        if self.const_std is None:
            s = x.std(dim=1)
            s = s.reshape(s.size()[0], 1)
        else:
            s = self.const_std
        """
            The normal cumulative distribution function is used to squash the values from within the range of 0 to 1
        """
        x = 0.5 * (1.0 + torch.erf((x - m) / (s * torch.sqrt(torch.tensor(
            2.0)))))
        x = x.reshape(s0, s1, s2)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
