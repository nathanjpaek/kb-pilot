import math
import torch
import torch.nn as nn
import torch.cuda
import torch.distributed


def get_activation_fn(activation):
    """Return an activation function Module according to its name."""
    if activation == 'gelu':
        fn = GELU()
    elif activation == 'relu':
        fn = nn.ReLU()
    elif activation == 'tanh':
        fn = nn.Tanh()
    else:
        raise ValueError(
            'Please pass a valid                           activation function'
            )
    return fn


class GELU(nn.Module):
    """ Implementation of the gelu activation function
        :cite:`DBLP:journals/corr/HendrycksG16`

        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi)
                    * (x + 0.044715 * torch.pow(x, 3))))

        Examples::
        >>> m = GELU()
        >>> inputs = torch.randn(2)
        >>> outputs = m(inputs)
    """

    def forward(self, x):
        gelu = x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        return gelu


class BertPredictionTransform(nn.Module):
    """{Linear(h,h), Activation, LN} block."""

    def __init__(self, hidden_size):
        """
        Args:
            hidden_size (int): BERT model hidden layer size.
        """
        super(BertPredictionTransform, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = get_activation_fn('gelu')
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states (Tensor): BERT encoder output ``(B, S, H)``
        """
        hidden_states = self.layer_norm(self.activation(self.dense(
            hidden_states)))
        return hidden_states


class MaskedLanguageModel(nn.Module):
    """predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size

    Args:
            hidden_size (int): output size of BERT model
            vocab_size (int): total vocab size
    """

    def __init__(self, hidden_size, vocab_size):
        super(MaskedLanguageModel, self).__init__()
        self.transform = BertPredictionTransform(hidden_size)
        self.decode = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x (Tensor): first output of bert encoder, ``(B, S, H)``
        Returns:
            prediction_log_prob (Tensor): shape ``(B, S, vocab)``
        """
        x = self.transform(x)
        prediction_scores = self.decode(x) + self.bias
        prediction_log_prob = self.log_softmax(prediction_scores)
        return prediction_log_prob


class NextSentencePrediction(nn.Module):
    """
    2-class classification model : is_next, is_random_next

    Args:
            hidden_size (int): BERT model output size
    """

    def __init__(self, hidden_size):
        super(NextSentencePrediction, self).__init__()
        self.linear = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x (Tensor): second output of bert encoder, ``(B, H)``
        Returns:
            seq_class_prob (Tensor): ``(B, 2)``
        """
        seq_relationship_score = self.linear(x)
        seq_class_log_prob = self.log_softmax(seq_relationship_score)
        return seq_class_log_prob


class BertPreTrainingHeads(nn.Module):
    """
    Bert Pretraining Heads: Masked Language Models, Next Sentence Prediction

    Args:
            hidden_size (int): output size of BERT model
            vocab_size (int): total vocab size
    """

    def __init__(self, hidden_size, vocab_size):
        super(BertPreTrainingHeads, self).__init__()
        self.next_sentence = NextSentencePrediction(hidden_size)
        self.mask_lm = MaskedLanguageModel(hidden_size, vocab_size)

    def forward(self, x, pooled_out):
        """
        Args:
            x (list of Tensor): all_encoder_layers, shape ``(B, S, H)``
            pooled_output (Tensor): second output of bert encoder, ``(B, H)``
        Returns:
            seq_class_log_prob (Tensor): next sentence prediction, ``(B, 2)``
            prediction_log_prob (Tensor): mlm prediction, ``(B, S, vocab)``
        """
        seq_class_log_prob = self.next_sentence(pooled_out)
        prediction_log_prob = self.mask_lm(x[-1])
        return seq_class_log_prob, prediction_log_prob


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4, 'vocab_size': 4}]
