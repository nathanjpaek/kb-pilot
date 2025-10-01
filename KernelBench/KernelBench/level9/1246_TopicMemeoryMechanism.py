import torch
import torch.multiprocessing
from torch import nn
import torch.utils.data


class TopicMemeoryMechanism(nn.Module):

    def __init__(self, topic_num, bow_size, embed_size):
        super(TopicMemeoryMechanism, self).__init__()
        self.topic_num = topic_num
        self.bow_size = bow_size
        self.embed_size = embed_size
        self.source_linear = nn.Linear(bow_size, embed_size)
        self.target_linear = nn.Linear(bow_size, embed_size)
        self.embed_project = nn.Linear(embed_size, embed_size)
        self.source_project = nn.Linear(embed_size, embed_size)
        self.weight_p = nn.Linear(embed_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, y_embed, topic_word_dist, topic_represent, gamma=0.8):
        batch_size = y_embed.shape[0]
        y_features = self.embed_project(y_embed)
        y_features = y_features.unsqueeze(1).expand(batch_size, self.
            topic_num, self.embed_size).contiguous()
        y_features = y_features.view(-1, self.embed_size)
        source_weight = self.relu(self.source_linear(topic_word_dist))
        source_features = self.source_project(source_weight)
        source_features = source_features.unsqueeze(0).expand(batch_size,
            self.topic_num, self.embed_size).contiguous()
        source_features = source_features.view(-1, self.embed_size)
        p_k_weights = self.sigmoid(self.weight_p(source_features + y_features))
        p_k_weights = p_k_weights.view(batch_size, self.topic_num)
        p_batch = torch.add(gamma * p_k_weights, topic_represent)
        target_weight = self.relu(self.target_linear(topic_word_dist))
        torch.matmul(p_batch, target_weight)
        return p_batch


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'topic_num': 4, 'bow_size': 4, 'embed_size': 4}]
