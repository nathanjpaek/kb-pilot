import torch
import torch.multiprocessing
from torch import nn
import torch.utils.data


class DocumentTopicDecoder(nn.Module):

    def __init__(self, dim_h, num_topics):
        super(DocumentTopicDecoder, self).__init__()
        self.decoder = nn.GRUCell(input_size=dim_h, hidden_size=dim_h)
        self.out_linear = nn.Linear(dim_h, num_topics)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, hidden):
        """
        Args:
            - input (bsz, dim_h)
            - hidden (bsz, dim_h)
            - avail_topic_mask (bsz, num_topics)
        Return:
            - hidden_out (bsz, dim_h) : hidden state of this step
            - topic_dist (bsz, num_topics) : probablity distribution of next sentence on topics
        """
        hidden_out = self.decoder(input, hidden)
        topic_dist = self.out_linear(hidden_out)
        topic_dist = self.softmax(topic_dist)
        return hidden_out, topic_dist


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'dim_h': 4, 'num_topics': 4}]
