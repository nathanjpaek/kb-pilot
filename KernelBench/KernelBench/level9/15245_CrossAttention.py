import torch


class CrossAttention(torch.nn.Module):
    """
    Implement of Co-attention.
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputA, inputB, maskA=None, maskB=None):
        """
        Input: embedding.
        """
        inputA.size(0)
        assert inputA.size(-1) == inputB.size(-1)
        scores = torch.bmm(inputA, inputB.transpose(1, 2))
        if maskA is not None and maskB is not None:
            maskA = maskA[:, :, None]
            maskB = maskB[:, None, :]
            mask = torch.bmm(maskA, maskB)
            scores = scores.masked_fill(mask == 0, -1000000000.0)
        attnA = torch.softmax(scores, 1)
        attnB = torch.softmax(scores, 2)
        cvA = torch.bmm(attnA.transpose(1, 2), inputA)
        cvB = torch.bmm(attnB, inputB)
        return cvA, cvB


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
