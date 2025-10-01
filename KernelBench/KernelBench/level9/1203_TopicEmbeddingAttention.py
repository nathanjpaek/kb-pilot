import torch
from torch.nn import functional as F
import torch.multiprocessing
from torch import nn
import torch.utils.data


class TopicEmbeddingAttention(nn.Module):
    """
    query： encoder的隐藏状态 key value：主题嵌入向量
    计算每个时间步t 对于加权topic embedding向量
    """

    def __init__(self, encoder_hidden_size, topic_num, topic_emb_dim):
        super(TopicEmbeddingAttention, self).__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.topic_num = topic_num
        self.topic_emb_dim = topic_emb_dim
        self.W = nn.Parameter(torch.Tensor(encoder_hidden_size, topic_emb_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, encoder_memory, topic_emb):
        """
            encoder_memory: [batch_size,seq_len,hidden_dim]
            attention_dist: [batch_size, seq_len]
            topic_emb:      [topic_num, embedding_dim]
            topic_dist:     [batch_size,topic_num]
        """
        encoder_memory.shape[0]
        topic_seq_w = torch.matmul(self.W, topic_emb.T)
        seq_topic_w = torch.matmul(encoder_memory, topic_seq_w)
        seq_topic_w = F.softmax(seq_topic_w, dim=2)
        hidden_topic_state = torch.matmul(seq_topic_w, topic_emb)
        return hidden_topic_state


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'encoder_hidden_size': 4, 'topic_num': 4, 'topic_emb_dim': 4}]
